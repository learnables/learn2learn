
#!/usr/bin/env python3

"""
Fetches the latest papers from the learn2learn spreadsheet and writes docs/source/paper_list.md

Requires:
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

from __future__ import print_function
import pickle
import os
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
#SAMPLE_SPREADSHEET_ID = '1uapV11T4-5sUy8QRAotUxaueGP2wErqoWodjQ957RpY'
#SAMPLE_RANGE_NAME = 'Form Responses 1!A2:E'
SAMPLE_SPREADSHEET_ID = '1Z9FOcQVNNnZ5vRXeC1x5dptjRmzE6WqlW-EK7YRwRdU'
SAMPLE_RANGE_NAME = 'Papers!A2:E'


def main():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
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

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
            range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])

    content = '\n'
    if not values:
        print('No data found.')
    else:
        for i, row in enumerate(reversed(values)):
            if len(row) >= 4:
                content += '**' + row[1] + '**' + '\n'
                content += '<br />' + '\n'
                content += 'by *' + row[2] + '*' + '\n'
                content += '<br />' + '\n'
                content += '[' + row[3] + '](' + row[3] + ')' + '\n'
                if len(row) >= 5:
                    content += '<br />' + '\n'
                    content += row[4] + '\n'
                content += '\n\n'
    header = """
# Paper List

The following papers were announced on the [learn2learn Twitter account](https://twitter.com/metalearn2learn). You can submit **unannounced** and **meta-learning related** papers through the following Google Form. (It does not matter if they are old or new, but they shouldn't be already announced.)

!!! info
    Announce any paper via the [Google Form to announce papers](https://docs.google.com/forms/d/e/1FAIpQLSeKPfYEttRKN3Lk317cxcNbU454yCTRktXpxMiK_O6PgFq22A/viewform?usp=sf_link), also available below.

## Submitted Papers

    """
    footer = """

## Submission Form

<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSeKPfYEttRKN3Lk317cxcNbU454yCTRktXpxMiK_O6PgFq22A/viewform?embedded=true" width="640" height="1200" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    """
    with open('docs/source/paper_list.md', 'w') as paper_file:
        paper_file.write(header)
        paper_file.write(content)
        paper_file.write(footer)


if __name__ == '__main__':
    main()
