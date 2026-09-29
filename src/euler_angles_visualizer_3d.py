"""
euler_angles_visualizer.py

Real-time OpenGL visualization of Euler Angles from BLE Orientation Data Service.

Displays:
  - Line 1: current absolute roll / pitch / yaw
  - Line 2: reference roll / pitch / yaw (set by pressing R)

The rectangular solid rotates to show the *relative* orientation
(current minus reference), matching the PyTeapot convention.

Press R   – capture current angles as the reference pose
Press ESC – quit
"""

import asyncio
import logging
import queue
import threading
import argparse
from typing import Optional

import pygame
from pygame.locals import OPENGL, DOUBLEBUF, QUIT, KEYDOWN, K_ESCAPE, K_r
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_LEQUAL, GL_PROJECTION, GL_MODELVIEW,
    GL_SMOOTH, GL_QUADS, GL_RGBA, GL_UNSIGNED_BYTE,
    GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST,
    glBegin, glEnd, glClear, glClearColor, glClearDepth,
    glColor3f, glDepthFunc, glEnable, glHint, glLoadIdentity,
    glMatrixMode, glRotatef, glShadeModel, glTranslatef,
    glVertex3f, glRasterPos3d, glDrawPixels, glViewport,
)
from OpenGL.GLU import gluPerspective

from ble_orientation_service import BLEOrientationService, OrientationData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 800, 480
FPS_CAP = 60

# ---------------------------------------------------------------------------
# BLE background thread helpers
# ---------------------------------------------------------------------------

_data_queue: queue.Queue[OrientationData] = queue.Queue()


def _on_ble_data(data: OrientationData) -> None:
    """BLE callback — runs on the background asyncio thread."""
    _data_queue.put_nowait(data)


async def _ble_loop_async(
    address: Optional[str],
    name: Optional[str],
    pair: bool,
    macos_use_bdaddr: bool,
    stop_event: asyncio.Event,
) -> None:
    """Async BLE work — runs entirely on the background thread."""
    service = BLEOrientationService(on_data_callback=_on_ble_data)
    try:
        device = await service.find_device(
            address=address,
            name=name,
            macos_use_bdaddr=macos_use_bdaddr,
        )
        if not device:
            logging.error("Failed to find BLE device.")
            return

        timeout = 60.0 if pair else 30.0
        if not await service.connect(device, pair=pair, timeout=timeout):
            logging.error("Failed to connect to BLE device.")
            return

        logging.info("Connected to BLE device.")

        # Poll until the main thread signals shutdown
        while not stop_event.is_set():
            await service.read_characteristics()
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logging.info("BLE loop cancelled.")
    except Exception as exc:
        logging.error("BLE loop error: %s", exc)
    finally:
        await service.disconnect()


def _run_ble_thread(
    address: Optional[str],
    name: Optional[str],
    pair: bool,
    macos_use_bdaddr: bool,
    stop_event: asyncio.Event,
) -> None:
    """Entry point for the dedicated BLE background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _ble_loop_async(address, name, pair, macos_use_bdaddr, stop_event)
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# OpenGL helpers
# ---------------------------------------------------------------------------

def _init_gl() -> None:
    glShadeModel(GL_SMOOTH)
    glClearColor(0.05, 0.05, 0.08, 1.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def _resize(width: int, height: int) -> None:
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def _draw_text(position: tuple, text: str, size: int) -> None:
    font = pygame.font.SysFont("Courier", size, bold=True)
    surface = font.render(text, True, (220, 220, 220, 255), (13, 13, 20, 255))
    data = pygame.image.tostring(surface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(surface.get_width(), surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, data)


def _draw_board(roll: float, pitch: float, yaw: float) -> None:
    """
    Draw a coloured rectangular solid rotated by (roll, pitch, yaw) in degrees.
    Rotation order: roll → Z, pitch → X, yaw → Y  (PyTeapot convention).
    """
    glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -7.0)

    glRotatef(-roll,  0.0, 0.0, 1.0)
    glRotatef( pitch, 1.0, 0.0, 0.0)
    glRotatef( yaw,   0.0, 1.0, 0.0)

    # Half-extents: wide (x), thin (y), moderate depth (z)
    hw, ht, hd = 1.8, 0.25, 1.1

    glBegin(GL_QUADS)

    # Top — green
    glColor3f(0.0, 0.85, 0.3)
    glVertex3f( hw,  ht, -hd)
    glVertex3f(-hw,  ht, -hd)
    glVertex3f(-hw,  ht,  hd)
    glVertex3f( hw,  ht,  hd)

    # Bottom — orange
    glColor3f(1.0, 0.45, 0.0)
    glVertex3f( hw, -ht,  hd)
    glVertex3f(-hw, -ht,  hd)
    glVertex3f(-hw, -ht, -hd)
    glVertex3f( hw, -ht, -hd)

    # Front — red
    glColor3f(0.9, 0.1, 0.1)
    glVertex3f( hw,  ht,  hd)
    glVertex3f(-hw,  ht,  hd)
    glVertex3f(-hw, -ht,  hd)
    glVertex3f( hw, -ht,  hd)

    # Back — yellow
    glColor3f(0.95, 0.85, 0.0)
    glVertex3f( hw, -ht, -hd)
    glVertex3f(-hw, -ht, -hd)
    glVertex3f(-hw,  ht, -hd)
    glVertex3f( hw,  ht, -hd)

    # Left — blue
    glColor3f(0.1, 0.35, 0.95)
    glVertex3f(-hw,  ht,  hd)
    glVertex3f(-hw,  ht, -hd)
    glVertex3f(-hw, -ht, -hd)
    glVertex3f(-hw, -ht,  hd)

    # Right — magenta
    glColor3f(0.85, 0.1, 0.85)
    glVertex3f( hw,  ht, -hd)
    glVertex3f( hw,  ht,  hd)
    glVertex3f( hw, -ht,  hd)
    glVertex3f( hw, -ht, -hd)

    glEnd()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def run(
    address: Optional[str],
    name: Optional[str],
    pair: bool,
    macos_use_bdaddr: bool,
    debug: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- Launch BLE background thread ---
    stop_event = asyncio.Event()
    ble_thread = threading.Thread(
        target=_run_ble_thread,
        args=(address, name, pair, macos_use_bdaddr, stop_event),
        daemon=True,
        name="ble-thread",
    )
    ble_thread.start()
    logging.info("BLE background thread started.")

    # --- pygame / OpenGL initialisation ---
    pygame.init()
    pygame.display.set_mode((WINDOW_W, WINDOW_H), OPENGL | DOUBLEBUF)
    pygame.display.set_caption("BLE Euler Angles — OpenGL Visualizer")
    _resize(WINDOW_W, WINDOW_H)
    _init_gl()
    clock = pygame.time.Clock()

    # Orientation state
    roll = pitch = yaw = 0.0
    ref_roll = ref_pitch = ref_yaw = 0.0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_r:
                        ref_roll, ref_pitch, ref_yaw = roll, pitch, yaw
                        logging.info(
                            "Reference set — Roll: %.2f°  Pitch: %.2f°  Yaw: %.2f°",
                            ref_roll, ref_pitch, ref_yaw,
                        )

            # Drain BLE queue (non-blocking)
            try:
                while True:
                    data = _data_queue.get_nowait()
                    if data.euler_angles:
                        roll, pitch, yaw = data.euler_angles
            except queue.Empty:
                pass

            # Relative angles drive the solid
            rel_roll  = roll  - ref_roll
            rel_pitch = pitch - ref_pitch
            rel_yaw   = yaw   - ref_yaw

            _draw_board(rel_roll, rel_pitch, rel_yaw)

            # Two HUD lines only
            _draw_text(
                (-2.6, 1.75, 2.0),
                f"Roll: {roll:+7.2f}   Pitch: {pitch:+7.2f}   Yaw: {yaw:+7.2f}",
                17,
            )
            _draw_text(
                (-2.6, 1.50, 2.0),
                f"Ref   {ref_roll:+7.2f}         {ref_pitch:+7.2f}         {ref_yaw:+7.2f}   [R]=set",
                17,
            )

            pygame.display.flip()
            clock.tick(FPS_CAP)

    finally:
        stop_event.set()
        ble_thread.join(timeout=5.0)
        pygame.quit()
        logging.info("Visualizer closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time OpenGL Euler Angles Visualizer from BLE Device"
    )
    parser.add_argument(
        "--address",
        type=str,
        help="BLE device address. On macOS use the CoreBluetooth UUID.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the BLE peripheral.",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Enable explicit pairing where supported.",
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

    run(
        address=args.address,
        name=args.name,
        pair=args.pair,
        macos_use_bdaddr=args.macos_use_bdaddr,
        debug=args.debug,
    )
