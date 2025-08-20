import os
from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary

class Library(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data=None, library=None):
        # Apply settings to environment for easy access in nodes
        try:
            settings = (library_data or {}).get("settings", [])
            for s in settings:
                contents = s.get("contents", {})
                base = contents.get("NIM_BASE_URL")
                key = contents.get("NIM_API_KEY")
                hf = contents.get("HUGGINGFACE_HUB_ACCESS_TOKEN") or contents.get("HF_TOKEN")
                if base:
                    os.environ.setdefault("NIM_BASE_URL", base)
                if key:
                    os.environ.setdefault("NIM_API_KEY", key)
                if hf:
                    os.environ.setdefault("HUGGINGFACE_HUB_ACCESS_TOKEN", hf)
                    os.environ.setdefault("HF_TOKEN", hf)
                break
        except Exception:
            pass

    def after_library_nodes_loaded(self, library_data=None, library=None):
        pass

# Export symbol expected by the engine
AdvancedLibrary = Library

