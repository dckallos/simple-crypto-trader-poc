"""
prompt_builder.py

A minimal module that:
1) Manages loading Mustache templates from a specified directory.
2) Renders them with a given context.
3) Returns the final prompt string for AIStrategy or other downstream use.

Dependencies:
    pip install pystache
"""

import os
import pystache
import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    A small class that loads and renders Mustache templates from the
    specified directory. The typical usage:
        builder = PromptBuilder(template_dir="templates")
        text = builder.render_template("aggregator_simple_prompt.mustache", context_dict)
    """

    def __init__(self, template_dir: str = "templates"):
        """
        :param template_dir: The directory where your Mustache files are stored.
        """
        self.template_dir = template_dir

    def render_template(self, template_name: str, context: dict) -> str:
        """
        Renders the specified Mustache template with the provided context.

        :param template_name: Filename of the mustache template, e.g. "aggregator_simple_prompt.mustache".
        :param context: A dictionary of data to be substituted into the template.
        :return: The fully rendered prompt string.
        """
        template_path = os.path.join(self.template_dir, template_name)

        if not os.path.isfile(template_path):
            logger.error(f"Template not found: {template_path}")
            raise FileNotFoundError(f"Could not locate template '{template_name}' in directory '{self.template_dir}'")

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
            rendered = pystache.render(template_str, context)
            return rendered
        except Exception as e:
            logger.exception(f"Error rendering template '{template_name}': {e}")
            raise
