import nltk
import string

# ======= Constants ========

# CODE_SNIPPET_PATTERN = r"(?:```|'''|\"\"\")(?:[\n\s]*\w+[\n\s]*)+(?:```|'''|\"\"\")"
CODE_SNIPPET_PATTERN = r"^```\n.*\n```$"

# JAVA_CODE_PATTERNS = {
#     "CLASS_DEF" : r'^\s*((public|private)\s*)?class\s+(\w+)(\s+extends\s+(\w+))?(\s+implements\s+([\w,\s]+))?\s+\{(\s*})?',
#     "FUNC_DEF" : r'^\s*\w+\s*\([^)]*\)\s*\{|(public|private|protected)\s+(static|final)?\s+(\w+)\s+(\w+)\([^)]*\)',
#     "IF" : r'^\s*if\s*\(.*\{|\w+\s*\([^)]*\)\s*\{',
#     "OBJ" : r'^\s*\w\s*=\snew[^;]+;',
#     "SPLIT_PUNCT" : r"^\s*([\"!#\\$%&()*+,-./:;<=>?@[\]^_'`{|}~])",
#     # "RVM_REPEATED_PUNC" : r"^\s*([%s])\\1{1,}" % string.punctuation,
#     "LOOP" : r"^\s*(for|while)\s*\([^)]*\)",
# }

SCALA_CODE_PATTERNS = {
    "CLASS_DEF" : r'^\s*(public|private)?\s+(class|interface|\w+)\s+(\w+)\s*({|\(.*?\))',
    "FUNC_DEF" : r'^\s*(public|private|protected)?\s+(static|final)?\s+(\w+)\s+(\w+)\([^)]*\)',
    "FUNC_CALL" : r"^\s*(?:.\w+(\(.*\))?)*$",
    "IF" : r'^\s*if\s*\(.*\{|\w+\s*\([^)]*\)\s*\{',
    "OBJ" : r'^\s*\b(\w+)\s+(\w+)\b|(val|var)\s+(\w+)\s*:(.+)',
    "SPLIT_PUNCT" : r"^([\"!#\\$%&()*+,-./:;<=>?@[\]^_'`{|}~])",
    # "RVM_REPEATED_PUNC" : "^([%s])\\1{1,}" % string.punctuation,
    "LOOP" : r"^\s*(for|while)\s*\([^)]*\)",
}

# JAVA_TOTAL_PATTERN = r"|".join(JAVA_CODE_PATTERNS.values())
JAVA_CODE_PATTERNS_WHOLE_LINE = {
    "end_w_semicolon" : r"^\s*.*;\s*$",
    "end_w_open_bracket" : r"^\s*.*({|\(|\[))\s*$",
    "func_chain" : r"^\s*(?:.\w+(\(.*\))?)*$",
    "close_bracket" : r"^\s*}$",
}

# PYTHON_CODE_PATTERN_WHOLE_LINE = {
#     "import" : r"^(from [a-zA-Z0-9]+ )?import \w+(?:,\s*[a-zA-Z0-9]+)*( as [a-zA-Z0-9]+)?$",
#     "class_func" : r"^\s*(class|def) [a-zA-Z0-9]+(\(.*\))?:\s*$",
#     "assignment" : r"^\s*[a-zA-Z0-9]+\s*=\s*(?:.\w+(\(.*\))?)+$",
#     "func_chain": r"^\s*(?:.\w+(\(.*\))?)*$",
# }



JAVA_TOTAL_PATTERN = r"|".join(JAVA_CODE_PATTERNS_WHOLE_LINE)
SCALA_TOTAL_PATTERN = r"|".join(SCALA_CODE_PATTERNS.values())
TOTAL_CODE_PATTERN = r"|".join([JAVA_TOTAL_PATTERN, SCALA_TOTAL_PATTERN])

LOGGING_PATTERNS = {
    "STARTS_W_DATETIME" : r"^\s*\d\d/\d\d/\d\d \d\d:\d\d:\d\d .*$",
    "STARTS_W_LOG" : r"^\s*(?:FATAL|ERROR|WARN|INFO|DEBUG|TRACE|\[(?:fatal|error|warn|info|debug|trace)\]) .*$",
    "STARTS_W_LOG_NUM" : r"^\s*\[\d+\] .*$",
    "STARTS_W_AT" : r"^\s*(?:at|Caused by:) .*$",
    "ERROR_EXCEPTION" : r"^\s*[^\s]*(?:Error|Exception):.*\n.*$",
    "STARTS_W_SCALA" : r"^\s*(?:scala|spark-sql)?>\s.*$",
    "STARTS_W_PYTHON" : r"^\s*>>>.*$",
    "DotDotDot" : r"^\s*... \d+ (?:more|elided)",
}

COMBINED_LOGGING_PATTERNS = r"|".join(LOGGING_PATTERNS.values())

# ======= Classes ========
class ClassicalPreprocessing:

    def __init__(self, tokenizer, stemmer, stopWords, filters=[]):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopWords = stopWords
        self.filters = filters

    def preprocess(self, text):
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)

        cleanTokens = []

        for token in tokens:
            for fil in self.filters:
                token = fil.filter(token, text)

            if stopwords:
                if token in self.stopWords:
                    continue

            if self.stemmer:
                token = self.stemmer.stem(token)

            if len(token) > 0:
                cleanTokens.append(token)

        return cleanTokens

# ======= Functions ========
def cleanDescription(desc):
    # Remove class declaration
    cleanDesc = re.sub(CLASS_DEF_JAVA_REGEX, '', desc)

    # Remove if
    cleanDesc = re.sub(IF_JAVA_REGEX, '', cleanDesc)

    # Remove function, catch and some ifs
    cleanDesc = re.sub(FUNC_IF_DEF_JAVA_REGEX, '', cleanDesc)

    # Remove variablie
    cleanDesc = re.sub(OBJ_JAVA_REGEX, '', cleanDesc)

    # Remove time
    cleanDesc = re.sub(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}', '', cleanDesc)

    # Remove date
    cleanDesc = re.sub(r'[0-9]{1,4}([/-])[0-9]{1,2}\1[0-9]{2,4}', '', cleanDesc)

    # Remove repeated punctuation like ######
    cleanDesc = re.sub(r"([,:;><!?=_\\\/*-.,])\1{1,}", '\\1', cleanDesc)

    newdesc = ""
    puncSet = set(string.punctuation)

    for l in cleanDesc.split("\n"):
        # Remove sentence that have less 10 characters
        if len(l) < 10:
            continue

        # Remove the majority of stack traces, some code and too short texts. Remove sentence that has less 5 tokens.
        nTok = 0
        for t in re.split(r'\s', l):
            if len(t) > 0:
                nTok += 1

        if nTok < 5:
            continue

        # Remove sentences which 20% of characters are numbers or punctuations
        npt = 0
        for c in l:
            if c.isnumeric() or c in puncSet:
                npt += 1

        if float(npt) / len(l) > 0.20:
            continue

        newdesc += l + '\n'

    return newdesc


def separate_code(text):
    # first we check if code snippet exists, if they exist, then just use them

    java_pattern = r"((public|private|protected)\s+(static|final)?\s+(\w+)\s+(\w+)\([^)]*\)|(class|interface)\s+(\w+)|(\w+)\s+(\w+)\b|//.*?$|/\*(.|\n)*?\*/|\"(?:\\.|[^\"\\])*\"|(if|for|while)\s*\([^)]*\)|\b\d+(\.\d+)?\b)"  # Comprehensive pattern for Java code elements
    code_blocks = re.findall(java_pattern, text, flags=re.DOTALL | re.MULTILINE)  # Match across multiple lines

    natural_text_parts = re.split(java_pattern, text, flags=re.DOTALL | re.MULTILINE)  # Split based on code blocks

    result = []
    for i in range(len(natural_text_parts)):
        if i % 2 == 0:  # Even-indexed parts are natural text
            result.append(natural_text_parts[i].strip())
        else:  # Odd-indexed parts are code blocks
            result.append({"code": code_blocks[i - 1]})  # Store code blocks as dictionaries

    return result