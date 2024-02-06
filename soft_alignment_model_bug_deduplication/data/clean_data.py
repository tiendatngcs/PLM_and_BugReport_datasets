import argparse
import codecs
import logging

import ujson

import sys

from classical_approach.generate_input_dbrd import DBRDPreprocessing
from data.bug_report_database import BugReportDatabase
from data.preprocessing import cleanDescription, softClean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_dataset', required=True,
                        help="File that contains all bug report data(categorical, summary,...)")
    parser.add_argument('--output', required=True,
                        help="Save clean file")
    parser.add_argument('--fields', nargs='+', help='A list of window sizes', required=True,
                        default=["description"])
    parser.add_argument('--type',
                        help='Option: agg (clean more aggressive the data), soft (put a space between words and punctuation, remove date)')
    parser.add_argument('--rm_punc', action="store_true")
    parser.add_argument('--rm_number', action="store_true")
    parser.add_argument('--sent_tok', action="store_true")
    parser.add_argument('--stop_words', action="store_true")
    parser.add_argument('--stem', action="store_true")
    parser.add_argument('--lower_case', action="store_true")
    parser.add_argument('--rm_char', action="store_true")

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logHandler = logging.StreamHandler()
    logger.addHandler(logHandler)

    logger.info("Loading bug reports content")
    bugReportDataset = BugReportDatabase.fromJson(args.bug_dataset)
    newFile = codecs.open(args.output, "w", encoding='utf-8')

    nEmptyBugs = 0

    if args.type == 'agg':
        logger.info("Aggressive clean")
        cleanFunc = cleanDescription
    elif args.type == 'soft':
        logger.info("Soft clean")
        cleanFunc = softClean
    elif args.type == 'rep':
        dbrd = DBRDPreprocessing()


        def clean_dbrd(text, rmPunc=False, sentTok=False, rmNumber=False):
            return " ".join(dbrd.preprocess(text))


        cleanFunc = clean_dbrd
    else:
        logger.error("Type argument unknown")
        sys.exit(-1)

    logger.info("Cleaning Data")
    for idx in range(len(bugReportDataset)):
        bug = bugReportDataset.getBugByIndex(idx)
        cleanBug = dict(bug)

        for fieldName in args.fields:
            if len(bug[fieldName]) == 0:
                continue

            l = len(bug[fieldName])

            if fieldName == 'description':
                cleanBug[fieldName] = cleanFunc(bug[fieldName], args.rm_punc, args.sent_tok, args.rm_number,
                                                args.stop_words, args.stem, args.lower_case, args.rm_char)
            else:
                cleanBug[fieldName] = cleanFunc(bug[fieldName], args.rm_punc, False, args.rm_number, args.stop_words,
                                                args.stem, args.lower_case, args.rm_char)

            if len(cleanBug[fieldName]) == 0:
                logger.info("Bug %s: %s content was erased!" % (bug['bug_id'], fieldName))
                nEmptyBugs += 1

        newFile.write(ujson.dumps(cleanBug))
        newFile.write("\n")

    logging.info("Total number of new empty bugs: %d" % nEmptyBugs)
    logging.info("Finish!!!")

