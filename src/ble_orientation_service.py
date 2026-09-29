"""
BLE Orientation Data Service Module
"""

import asyncio
import logging
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

logger = logging.getLogger(__name__)


@dataclass
class OrientationData:
    """Container for orientation sensor data."""

    timestamp: Optional[int] = None
    quaternions: Optional[tuple] = None  # (x, y, z, w)
    linear_acceleration: Optional[tuple] = None  # (x, y, z)
    euler_angles: Optional[tuple] = None  # (roll, pitch, yaw)


class BLEOrientationService:
    """Manages BLE connection and reads orientation data characteristics."""

    ORIENTATION_DATA_UUIDS = {
        "timestamp": str(UUID("C931138E-D47A-4594-B236-AB3CB8D46D4D")).lower(),
        "quaternions": str(UUID("CC9B0D63-99F4-47AF-8AAC-2CC88846BB2C")).lower(),
        "linear_acceleration": str(
            UUID("CC9B0D63-99F4-47AF-8AAC-2CC88846BB3C")
        ).lower(),
        "euler_angles": str(UUID("f3a38005-66ac-4ec8-9bab-30e77ac32ae8")).lower(),
    }

    def __init__(
        self, on_data_callback: Optional[Callable[[OrientationData], None]] = None
    ):
        """
        Initialize the BLE Orientation Service.

        Args:
            on_data_callback: Optional callback function called when new data is received.
                             Receives an OrientationData object.
        """
        self.on_data_callback = on_data_callback
        self.current_data = OrientationData()
        self.is_connected = False
        self.client: Optional[BleakClient] = None

    @staticmethod
    def _scanner_kwargs(macos_use_bdaddr: bool) -> dict:
        return (
            {"cb": {"use_bdaddr": True}}
            if sys.platform == "darwin" and macos_use_bdaddr
            else {}
        )

    @staticmethod
    def _corebluetooth_uuid(device: BLEDevice) -> Optional[str]:
        if sys.platform != "darwin":
            return None

        try:
            peripheral = device.details[0]
            return str(peripheral.identifier().UUIDString())
        except Exception:
            return None

    @staticmethod
    def _format_advertisement(device: BLEDevice, adv: AdvertisementData) -> str:
        parts = [
            f"address={device.address}",
            f"name={device.name or 'unknown'}",
            f"adv_name={adv.local_name or 'unknown'}",
            f"rssi={adv.rssi}",
        ]

        cb_uuid = BLEOrientationService._corebluetooth_uuid(device)
        if cb_uuid:
            parts.append(f"corebluetooth_uuid={cb_uuid}")

        if adv.service_uuids:
            parts.append("services=" + ",".join(adv.service_uuids))
        else:
            parts.append("services=none")

        return " | ".join(parts)

    async def scan_devices(
        self, timeout: float = 10.0, macos_use_bdaddr: bool = False
    ) -> list[tuple[BLEDevice, AdvertisementData]]:
        """Scan for nearby BLE devices and log compact advertisement details."""
        logger.info("Scanning for BLE devices for %.1f seconds...", timeout)
        results = await BleakScanner.discover(
            timeout=timeout,
            return_adv=True,
            **self._scanner_kwargs(macos_use_bdaddr),
        )

        devices = list(results.values())
        if not devices:
            logger.warning("No BLE devices discovered.")
            return []

        for index, (device, adv) in enumerate(devices, start=1):
            logger.info(
                "Scan result %s: %s", index, self._format_advertisement(device, adv)
            )

        return devices

    async def find_device(
        self,
        address: Optional[str] = None,
        name: Optional[str] = None,
        macos_use_bdaddr: bool = False,
    ) -> Optional[BLEDevice]:
        """Find a BLE device by address or name."""
        scanner_kwargs = self._scanner_kwargs(macos_use_bdaddr)

        if address:
            logger.info(f"Searching for device with address: {address}")
            if sys.platform == "darwin" and ":" in address and not macos_use_bdaddr:
                logger.warning(
                    "macOS CoreBluetooth does not expose BLE MAC addresses. "
                    "Use the CoreBluetooth UUID shown in scan results, search by --name, "
                    "or pass --macos-use-bdaddr to try Bleak's undocumented MAC-address workaround."
                )
            elif sys.platform == "darwin" and macos_use_bdaddr:
                logger.warning(
                    "Using Bleak's undocumented macOS Bluetooth-address lookup. "
                    "If discovery is unreliable, use the CoreBluetooth UUID or --name instead."
                )

            device = await BleakScanner.find_device_by_address(
                address, timeout=20.0, **scanner_kwargs
            )

            if device:
                logger.info(f"Found device by address: {device}")
                cb_uuid = self._corebluetooth_uuid(device)
                if cb_uuid:
                    logger.info("CoreBluetooth UUID for device: %s", cb_uuid)
                return device

            # Fallback: scan all devices
            logger.warning("Direct address lookup failed. Scanning all devices...")
            scanner = BleakScanner(**scanner_kwargs)
            devices = await scanner.discover(timeout=20.0)

            target_address = address.lower().replace("-", ":").replace("_", ":")
            for device in devices:
                device_address = (
                    device.address.lower().replace("-", ":").replace("_", ":")
                )
                if device_address == target_address:
                    logger.info(f"Found device by scan: {device}")
                    cb_uuid = self._corebluetooth_uuid(device)
                    if cb_uuid:
                        logger.info("CoreBluetooth UUID for device: %s", cb_uuid)
                    return device

            logger.error(f"Device {address} not found.")
            if devices:
                logger.info(
                    "Available devices: "
                    + ", ".join(
                        f"{d.address} ({d.name or 'unknown name'})" for d in devices
                    )
                )

            if name:
                logger.info(f"Falling back to device name lookup: {name}")
            else:
                return None

        if name:
            logger.info(f"Searching for device with name: {name}")
            device = await BleakScanner.find_device_by_name(
                name, timeout=20.0, **scanner_kwargs
            )
            if device:
                logger.info(f"Found device by name: {device}")
                cb_uuid = self._corebluetooth_uuid(device)
                if cb_uuid:
                    logger.info("CoreBluetooth UUID for device: %s", cb_uuid)
            else:
                logger.error(f"Device {name} not found.")
            return device

        logger.error("Either address or name must be provided.")
        return None

    async def connect(
        self,
        device: BLEDevice,
        pair: bool = True,
        timeout: float = 30.0,
        max_retries: int = 3,
        services: Optional[list[str]] = None,
    ) -> bool:
        """Connect to a BLE device with automatic retry logic."""
        effective_pair = pair
        if pair and sys.platform == "darwin":
            logger.warning(
                "Explicit BLE pairing is not available on macOS. "
                "CoreBluetooth prompts for pairing only when an authenticated "
                "characteristic is accessed."
            )
            effective_pair = False

        for attempt in range(max_retries):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{max_retries}...")
                await self.disconnect()

                self.client = BleakClient(
                    device,
                    disconnected_callback=self._on_client_disconnected,
                    services=services,
                    pair=effective_pair,
                    timeout=timeout,
                )
                await self.client.connect()

                if not self.client.is_connected:
                    raise RuntimeError(
                        "Bleak connect completed, but client is not connected"
                    )

                self.is_connected = True
                logger.info(
                    f"Connected to {device.name or 'unknown device'} ({device.address})"
                )
                self._log_orientation_characteristics()
                return True
            except Exception as e:
                self.is_connected = False
                await self.disconnect()
                logger.warning(
                    "Connection attempt %s failed: %s: %r",
                    attempt + 1,
                    type(e).__name__,
                    e,
                )
                if attempt < max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(2.0)
                else:
                    logger.error(
                        "Failed to connect after %s attempts. On macOS, try --name "
                        "instead of a hardware MAC address and do not rely on --pair.",
                        max_retries,
                    )
                    return False
        return False

    async def disconnect(self) -> None:
        """Disconnect from BLE device."""
        client = self.client
        self.client = None
        self.is_connected = False

        if client and client.is_connected:
            await client.disconnect()
            logger.info("Disconnected from device.")

    def _on_client_disconnected(self, _: BleakClient) -> None:
        """Track unsolicited peripheral disconnects."""
        self.is_connected = False
        logger.warning("BLE device disconnected.")

    def _log_orientation_characteristics(self) -> None:
        """Log whether the expected orientation characteristics were discovered."""
        if not self.client:
            return

        discovered = {
            characteristic.uuid.lower()
            for service in self.client.services
            for characteristic in service.characteristics
        }
        expected = set(self.ORIENTATION_DATA_UUIDS.values())
        missing = expected - discovered

        if missing:
            logger.warning(
                "Connected, but missing orientation characteristics: %s",
                ", ".join(sorted(missing)),
            )
        else:
            logger.info("Discovered all orientation characteristics.")

    async def read_characteristics(self) -> Optional[OrientationData]:
        """Read all orientation characteristics once. Returns updated OrientationData."""
        if not self.client or not self.client.is_connected:
            self.is_connected = False
            logger.error("Not connected to device.")
            return None

        try:
            for service in self.client.services:
                for characteristic in service.characteristics:
                    char_uuid = characteristic.uuid.lower()

                    if char_uuid not in self.ORIENTATION_DATA_UUIDS.values():
                        continue

                    if "read" not in str(characteristic.properties).lower():
                        continue

                    try:
                        value = await asyncio.wait_for(
                            self.client.read_gatt_char(characteristic.uuid), timeout=5.0
                        )

                        self._parse_characteristic(char_uuid, value)

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout reading {char_uuid}")
                    except Exception as e:
                        error_str = str(e).lower()
                        if "offset is invalid" not in error_str:
                            logger.debug(f"Read error for {char_uuid}: {e}")

            if self.on_data_callback:
                self.on_data_callback(self.current_data)

            return self.current_data

        except Exception as e:
            logger.error(f"Error reading characteristics: {e}")
            return None

    def _parse_characteristic(self, char_uuid: str, value: bytes) -> None:
        """Parse characteristic value and update current_data."""
        try:
            if char_uuid == self.ORIENTATION_DATA_UUIDS["timestamp"]:
                if len(value) == 8:
                    (ts,) = struct.unpack("<Q", value)
                    self.current_data.timestamp = ts
                    logger.debug(f"Timestamp: {ts}")

            elif char_uuid == self.ORIENTATION_DATA_UUIDS["quaternions"]:
                if len(value) == 16:
                    x, y, z, w = struct.unpack("<ffff", value)
                    self.current_data.quaternions = (x, y, z, w)
                    logger.debug(f"Quaternions: ({x}, {y}, {z}, {w})")

            elif char_uuid == self.ORIENTATION_DATA_UUIDS["linear_acceleration"]:
                if len(value) == 12:
                    x, y, z = struct.unpack("<fff", value)
                    self.current_data.linear_acceleration = (x, y, z)
                    logger.debug(f"Linear Acceleration: ({x}, {y}, {z})")

            elif char_uuid == self.ORIENTATION_DATA_UUIDS["euler_angles"]:
                if len(value) == 12:
                    roll, pitch, yaw = struct.unpack("<fff", value)
                    self.current_data.euler_angles = (roll, pitch, yaw)
                    logger.debug(f"Euler Angles: ({roll}, {pitch}, {yaw})")

        except Exception as e:
            logger.error(f"Error parsing characteristic {char_uuid}: {e}")

    async def start_polling(self, interval: float = 0.0) -> None:
        """Start continuously polling orientation data."""
        if interval > 0:
            logger.info(f"Starting polling with {interval}s interval...")
        else:
            logger.info("Starting continuous polling with no fixed delay...")
        try:
            while self.client and self.client.is_connected:
                self.is_connected = True
                await self.read_characteristics()
                if interval > 0:
                    await asyncio.sleep(interval)
                else:
                    await asyncio.sleep(0)
            self.is_connected = False
        except asyncio.CancelledError:
            logger.info("Polling cancelled.")
        except Exception as e:
            self.is_connected = False
            logger.error(f"Error in polling loop: {e}")
