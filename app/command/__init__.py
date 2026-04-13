"""
Command layer: Change objects, Command protocol, CommandRegistry, CommandExecutor.

Application services produce Change objects; CommandRegistry turns them into
Command objects (with undo state); CommandExecutor applies changes and
manages undo/redo via UndoRedoStack.
"""
