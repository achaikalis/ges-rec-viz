"""euler_angles_visualizer.py

Real-time GUI visualization of Euler Angles from BLE Orientation Data Service.
"""

import argparse
import asyncio
import logging
import queue
import threading
import tkinter as tk
from typing import Optional

from ble_orientation_service import BLEOrientationService, OrientationData


class EulerAnglesVisualizer:
    """GUI application for visualizing Euler angles from BLE device."""

    def __init__(
        self,
        address: Optional[str] = None,
        name: Optional[str] = None,
        pair: bool = False,
        macos_use_bdaddr: bool = False,
        debug: bool = False,
    ):
        self.address = address
        self.name = name
        self.pair = pair
        self.macos_use_bdaddr = macos_use_bdaddr
        self.debug = debug

        self.service = BLEOrientationService(on_data_callback=self._on_ble_data)

        # Reference pose for relative angle calculation
        self.reference_roll = 0.0
        self.reference_pitch = 0.0
        self.reference_yaw = 0.0

        # Current angles
        self.current_roll = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0

        # Thread-safe queue for BLE → Tk data transfer
        self._data_queue: queue.Queue[OrientationData] = queue.Queue()
        self._ble_thread: Optional[threading.Thread] = None
        self._ble_loop: Optional[asyncio.AbstractEventLoop] = None
        self.closing = False

        # GUI setup
        self.root = tk.Tk()
        self.root.title("Gesture Recognition – Demo")
        self.root.resizable(False, False)
        self.root.geometry("1280x250")
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        """Setup UI components."""
        # Zeroed angle display
        self.euler_var = tk.StringVar(value="Roll: –  Pitch: –  Yaw: –")
        tk.Label(
            self.root,
            textvariable=self.euler_var,
            font=("Courier New", 28, "bold"),
            bg="lightblue",
            pady=20,
        ).pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 5))

        # Reference display
        self.ref_var = tk.StringVar(
            value="Reference: Roll=0.00°  Pitch=0.00°  Yaw=0.00°"
        )
        tk.Label(
            self.root,
            textvariable=self.ref_var,
            font=("Courier New", 12),
            fg="grey",
        ).pack(padx=10, pady=5)

        # Connection status
        self.status_var = tk.StringVar(value="Status: Connecting...")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Courier New", 11),
            fg="darkgreen",
        ).pack(padx=10, pady=5)

        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(padx=10, pady=(10, 10), fill=tk.X)
        tk.Button(
            button_frame,
            text="⟳ Zero / Set Reference",
            font=("Courier New", 14),
            command=self._zero_reference,
            bg="lightgreen",
            width=20,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame,
            text="Exit",
            font=("Courier New", 14),
            command=self._on_exit,
            bg="lightcoral",
            width=10,
        ).pack(side=tk.RIGHT, padx=5)

    def _zero_reference(self) -> None:
        """Set current orientation as the reference pose."""
        # FIX: capture the raw baseline first, then display exactly those values.
        self.reference_roll = self.current_roll
        self.reference_pitch = self.current_pitch
        self.reference_yaw = self.current_yaw
        self.ref_var.set(
            f"Reference: Roll={self.reference_roll:+.2f}°  "
            f"Pitch={self.reference_pitch:+.2f}°  "
            f"Yaw={self.reference_yaw:+.2f}°"
        )
        logging.info(
            f"Reference set at raw baseline: R={self.reference_roll:.2f}° "
            f"P={self.reference_pitch:.2f}° Y={self.reference_yaw:.2f}°"
        )
        self._update_readings()

    # ------------------------------------------------------------------
    # Tk-side update loops
    # ------------------------------------------------------------------
    def _drain_queue(self) -> None:
        """
        Drain the thread-safe data queue on the Tk thread.
        Called frequently via root.after(). Consumes all pending
        OrientationData frames produced by the BLE background thread,
        updating current_* attributes for the display loop to read.
        """
        try:
            while True:
                data = self._data_queue.get_nowait()
                if data.euler_angles:
                    self.current_roll, self.current_pitch, self.current_yaw = (
                        data.euler_angles
                    )
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(20, self._drain_queue)

    @staticmethod
    def _wrap(angle: float) -> float:
        """FIX: add angle wrapping functionality for normalizing an angle to (-180, 180]."""
        return (angle + 180.0) % 360.0 - 180.0

    def _relative_angles(self) -> tuple[float, float, float]:
        """Return current Euler readings offset by the captured reference."""
        # FIX: wrap each difference — (current - reference) can legitimately
        # exceed ±180°, which previously displayed as e.g. +358° instead of -2°.
        return (
            self._wrap(self.current_roll - self.reference_roll),
            self._wrap(self.current_pitch - self.reference_pitch),
            self._wrap(self.current_yaw - self.reference_yaw),
        )

    def _update_readings(self) -> None:
        rel_roll, rel_pitch, rel_yaw = self._relative_angles()
        self.euler_var.set(
            f"Roll: {rel_roll:+.2f}°  "
            f"Pitch: {rel_pitch:+.2f}°  "
            f"Yaw: {rel_yaw:+.2f}°"
        )

    def _update_display(self) -> None:
        """Refresh GUI labels with the latest orientation data."""
        try:
            self._update_readings()
            status = "Connected" if self.service.is_connected else "Disconnected"
            self.status_var.set(f"Status: {status}")
        except Exception as e:
            logging.error(f"Error updating display: {e}")
        if not self.closing:
            self.root.after(20, self._update_display)

    # ------------------------------------------------------------------
    # BLE background thread
    # ------------------------------------------------------------------
    def _on_ble_data(self, data: OrientationData) -> None:
        """
        BLE callback — runs on the background asyncio thread.
        Enqueues the received OrientationData frame for consumption
        by _drain_queue() on the Tk thread, avoiding any direct
        cross-thread Tk widget access.
        """
        self._data_queue.put_nowait(data)

    def _run_ble_thread(self) -> None:
        """Entry point for the BLE background thread."""
        self._ble_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ble_loop)
        try:
            self._ble_loop.run_until_complete(self._ble_loop_async())
        finally:
            self._ble_loop.close()

    async def _ble_loop_async(self) -> None:
        """Async BLE work — runs entirely on the background thread."""
        try:
            device = await self.service.find_device(
                address=self.address,
                name=self.name,
                macos_use_bdaddr=self.macos_use_bdaddr,
            )
            if not device:
                logging.error("Failed to find device.")
                return
            timeout = 60.0 if self.pair else 30.0
            if not await self.service.connect(device, pair=self.pair, timeout=timeout):
                logging.error("Failed to connect to device.")
                return
            logging.info("Connected to BLE device.")
            await self.service.start_polling()
        except asyncio.CancelledError:
            logging.info("BLE loop cancelled.")
        except Exception as e:
            logging.error(f"BLE loop error: {e}")
        finally:
            await self.service.disconnect()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _shutdown_ble(self) -> None:
        """Signal the BLE thread to stop and block until it joins."""
        if self._ble_loop and not self._ble_loop.is_closed():
            self._ble_loop.call_soon_threadsafe(self._ble_loop.stop)
        if self._ble_thread and self._ble_thread.is_alive():
            self._ble_thread.join(timeout=5.0)

    def _on_exit(self) -> None:
        """Handle application exit from button press or window close."""
        if self.closing:
            return
        logging.info("Shutting down...")
        self.closing = True
        self._shutdown_ble()
        self.root.destroy()

    def run(self) -> None:
        """Start the GUI application."""
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        # Launch BLE work on a dedicated background thread so that
        # CoreBluetooth delegate callbacks are never starved by Tk's
        # main-thread event dispatch.
        self._ble_thread = threading.Thread(
            target=self._run_ble_thread, daemon=True, name="ble-thread"
        )
        self._ble_thread.start()
        logging.info("BLE background thread started.")

        # Start Tk-side periodic loops
        self._drain_queue()
        self._update_display()
        try:
            self.root.mainloop()
        finally:
            if not self.closing:
                self.closing = True
                self._shutdown_ble()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Euler Angles Visualizer from BLE Device"
    )
    parser.add_argument(
        "--address",
        type=str,
        help="BLE device address. On macOS this is the CoreBluetooth UUID, not the MAC address.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of BLE device.",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Enable explicit pairing where supported. macOS pairs implicitly when needed.",
    )
    parser.add_argument(
        "--macos-use-bdaddr",
        action="store_true",
        help="On macOS, try Bleak's undocumented hardware MAC-address lookup.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose BLE/debug logging.",
    )
    args = parser.parse_args()
    if not args.address and not args.name:
        parser.error("Either --address or --name must be provided.")
    app = EulerAnglesVisualizer(
        address=args.address,
        name=args.name,
        pair=args.pair,
        macos_use_bdaddr=args.macos_use_bdaddr,
        debug=args.debug,
    )
    app.run()