"""Standalone operational scripts (Phase 11).

These are run directly (``python scripts/<name>.py``), not imported by the
package. Pure helpers live at module top level so they are unit-testable on CPU;
heavy/GPU and network imports stay inside ``main``/runtime functions.
"""
