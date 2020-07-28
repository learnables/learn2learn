#!/usr/bin/env python3

"""
Processes a bibtex file and adds its content to a Google Sheet.

Usage: python scripts/bibtex_to_gsheets.py mybibtext.tex

Requires: pip install bibtexparser
"""

import os
import sys
import pickle
import datetime
import bibtexparser

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1Z9FOcQVNNnZ5vRXeC1x5dptjRmzE6WqlW-EK7YRwRdU'
SAMPLE_RANGE_NAME = 'Papers!A2:E'


def google_sheets_login():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                    os.path.expanduser('scripts/credentials.json'), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    return service


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Please only pass the path to bib.'

    # Parse bibtex file
    bib_path = sys.argv[-1]
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    with open(bib_path) as bib_file:
        bib_db = bibtexparser.load(bib_file, parser=parser)

    # Parse existing papers
    service = google_sheets_login()
    sheet = service.spreadsheets()
    result = sheet.values().get(
        spreadsheetId=SAMPLE_SPREADSHEET_ID,
        range=SAMPLE_RANGE_NAME
    ).execute()
    existing_rows = result.get('values', [])
    all_titles = [row[1] for row in existing_rows if len(row) >= 2]

    new_rows = []
    for entry in reversed(bib_db.entries):
        # Insert into DB if title is not duplicate
        title = entry.get('title', None)
        title = title.replace('{', '').replace('}', '')
        if title is not None and title not in all_titles:
            authors = entry.get('author', 'Unknown')
            link = entry.get('url', 'http://learn2learn.net')
            summary = entry.get('annote', '')
            print('Title:', title)
            print('Authors:', authors)
            print('Link:', link)
            print('Summary:', summary)
            print('')
            row = [str(datetime.datetime.now()), title, authors, link, summary]
            new_rows.append(row)

    # Send to spreadsheet
    body = {
        'values': new_rows,
    }
    result = service.spreadsheets().values().append(
        spreadsheetId=SAMPLE_SPREADSHEET_ID,
        range=SAMPLE_RANGE_NAME,
        valueInputOption='RAW',
        body=body
    ).execute()
    print('Added', len(new_rows), '/', len(bib_db.entries), 'new entries')
