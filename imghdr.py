# Minimal imghdr compatibility layer for python-telegram-bot on Python 3.13+
# This replicates the old stdlib imghdr.what() behavior for common formats.

import struct

def what(file, h=None):
    """
    Roughly matches the old imghdr.what() API:
    - file: filename OR file-like object
    - h: optional bytes header
    Returns a string like 'jpeg', 'png', 'gif', etc., or None.
    """
    if h is None:
        if hasattr(file, "read"):
            # file-like object
            pos = file.tell()
            h = file.read(32)
            file.seek(pos)
        else:
            # filename
            with open(file, "rb") as f:
                h = f.read(32)

    # PNG
    if h.startswith(b"\211PNG\r\n\032\n"):
        return "png"

    # GIF
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"

    # JPEG
    if len(h) >= 3 and h[0:3] == b"\xff\xd8\xff":
        return "jpeg"

    # BMP
    if h.startswith(b"BM"):
        return "bmp"

    # WebP (RIFF....WEBP)
    if len(h) >= 12 and h[0:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "webp"

    # TIFF (II*\x00 or MM\x00*)
    if h[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"

    # ICO
    if len(h) >= 4:
        marker = struct.unpack("<H", h[0:2])[0]
        imgcount = struct.unpack("<H", h[2:4])[0]
        # very rough check for .ico header
        if marker == 0 and imgcount > 0:
            return "ico"

    return None
