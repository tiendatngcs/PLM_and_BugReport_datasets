import nltk
import string

# ======= Constants ========
SCALA_CODE_PATTERNS = {
    "VAR_DEC" : r"val\s+\w+\s*=\s*.+",
    "FUNC_DEC" : r"(def|val|var)\s+\w+\s*\(.*\)\s*:\s*.+",
    "CLASS_DEC" : r"(class|trait|object)\s+\w+\s*.+",
    "IMPORT" : r"import\s+.+",
    "COMMENT" : r"//.*|/\*[\s\S]*?\*/",
    "IF" : r"";
}

JAVA_CODE_PATTERNS = {
    "VAR_DEC" : r"(public|private|protected)?\s*(static)?\s*(final)?\s*(void|int|char|boolean|float|double|long|short|byte)\s+(\w+)\s*(=[^;]+)?;",
    "FUNC_DEC" : r'(public|private|protected)?\s*(static)?\s*(void|int|char|boolean|float|double|long|short|byte)\s+\w+\s*\([^)]*\)\s*',
    "CLASS_DEC" : r"(public|private|protected)?\s*(abstract|final)?\s*(class|interface)\s+\w+\s*",
    "IMPORT" : r"import\s+.+",
    "COMMENT" : r"//.*|/\*[\s\S]*?\*/",
    "IF" : r'if\s*\(.*\{',
}



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
