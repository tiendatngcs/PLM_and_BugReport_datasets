# from preprocessing import ClassicalPreprocessing
from preprocess_text import *
import re

class ComponentAutoExtractor:
    def __init__(self, text, preprocess=False):
        self._text = text

    def _block_is_log(self, text_block):
        m = re.search(COMBINED_LOGGING_PATTERNS, text_block)
        return m is not None
    
    def has_code(self,):
        codes = re.findall(TOTAL_CODE_PATTERN, self._text, flags=re.DOTALL | re.MULTILINE)
        return len(codes) != 0

    def has_log(self,):
        logs = re.findall(COMBINED_LOGGING_PATTERNS, self._text, flags=re.DOTALL | re.MULTILINE)
        return len(logs) != 0

    def extract_code_snippet(self,):
        text = self._text
        # check if has code snippet pattern
        match = re.search(CODE_SNIPPET_PATTERN, text)
        if match:
            print("Fix this func")
            assert(False)
            # extract based on matched pattern then terminate
            block = re.findall(CODE_SNIPPET_PATTERN, text, flags=re.DOTALL | re.MULTILINE)  # Match across multiple lines
            remainder = re.split(CODE_SNIPPET_PATTERN, text, flags=re.DOTALL | re.MULTILINE)  # Split based on code blocks
            # now block can be code block or log block
            if not self._block_is_log(block):
                return block, remainder
            # fall through if is log

        # otherwise, loop through each line and extract
        code_part = ""
        remainder = ""
        for line in text.split("\n"):
            c = re.findall(TOTAL_CODE_PATTERN, text, flags=re.DOTALL | re.MULTILINE)  # Match across multiple lines
            r =  re.split(TOTAL_CODE_PATTERN, text, flags=re.DOTALL | re.MULTILINE)  # Split based on code blocks
            code_part += c + "\n"
            remainder += r + "\n"
        
        return code_part, remainder

    def extract_log(self,):
        text = self._text
        # check if has code snippet pattern
        match = re.search(CODE_SNIPPET_PATTERN, text)
        if match:
            # extract based on matched pattern then terminate
            block = "\n".join(re.findall(CODE_SNIPPET_PATTERN, text, flags=re.DOTALL | re.MULTILINE))  # Match across multiple lines
            remainder = "\n".join(re.split(CODE_SNIPPET_PATTERN, text, flags=re.DOTALL | re.MULTILINE))  # Split based on code blocks
            # now block can be code block or log block
            if self._block_is_log(block):
                return block, remainder
            # fall through if is not log

        # loop through each line and extract
        log_part = ""
        remainder = ""
        # for line in text.split("\n"):
        # not log
        log_part += "\n".join(re.findall(COMBINED_LOGGING_PATTERNS, text, flags=re.MULTILINE))
        # is log
        remainder += "\n".join(re.split(COMBINED_LOGGING_PATTERNS, text, flags=re.MULTILINE))

        return log_part, remainder

