"""Monkey patch to fix transformers additional_chat_templates bug"""
import transformers.utils.hub as hub

original_list_repo_templates = hub.list_repo_templates

def patched_list_repo_templates(*args, **kwargs):
    try:
        return original_list_repo_templates(*args, **kwargs)
    except Exception:
        # Return empty list if additional_chat_templates folder doesn't exist
        return []

hub.list_repo_templates = patched_list_repo_templates
