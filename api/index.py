from flask import Flask
from werkzeug.wrappers import Request, Response

# Create the Flask app
app = Flask(__name__)

# Define your routes
@app.route('/', methods=['GET'])
def home():
    return "Hello from Flask on Vercel!"

# Add more routes as needed

# Vercel serverless function handler
def handler(event, context):
    # Create a WSGI environment from the Vercel event
    environ = {
        'REQUEST_METHOD': event['httpMethod'],
        'SCRIPT_NAME': '',
        'PATH_INFO': event['path'],
        'QUERY_STRING': '&'.join([f"{k}={v}" for k, v in event.get('queryStringParameters', {}).items()]),
        'CONTENT_TYPE': event.get('headers', {}).get('content-type', ''),
        'CONTENT_LENGTH': event.get('headers', {}).get('content-length', ''),
        'SERVER_NAME': 'vercel',
        'SERVER_PORT': '443',
        'SERVER_PROTOCOL': 'HTTP/1.1',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https',
        'wsgi.input': event.get('body', '').encode('utf-8'),
        'wsgi.errors': None,
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
    }
    headers = event.get('headers', {})
    for key, value in headers.items():
        environ[f'HTTP_{key.upper().replace("-", "_")}'] = value

    # Prepare the response
    response_headers = []
    status = [200]

    def start_response(status_line, headers_list):
        status[0] = int(status_line.split()[0])
        response_headers.extend(headers_list)

    # Call the Flask app with the WSGI environment
    response_data = app(environ, start_response)
    response_body = b''.join(response_data).decode('utf-8')

    # Return the response in Vercel's expected format
    return {
        'statusCode': status[0],
        'headers': dict(response_headers),
        'body': response_body
    }