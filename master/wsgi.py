"""Waitress entry point for the master systemd service."""

from master.app import create_app

app = create_app()
