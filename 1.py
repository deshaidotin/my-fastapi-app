
import os
import re
import math
import requests
from flask import Flask, request, Response, jsonify, render_template
from flask_cors import CORS
import yt_dlp

# Tell Flask to serve HTML and static files directly from the 'static' folder
app = Flask(__name__, template_folder='static', static_folder='static')

# Enable CORS across all endpoints
CORS(app)

def clean_filename(title):
    """Sanitize string to be safe for HTTP headers and file names."""
    return re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

def format_bytes(size_bytes):
    """Convert raw byte size to readable MB string."""
    if not size_bytes or size_bytes == 0:
        return "Calculated on Stream"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

@app.route('/', methods=['GET'])
def home():
    """Serves index.html from the 'static' directory when root URL is requested."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend server status."""
    return jsonify({
        'status': 'online', 
        'message': 'MediaFetch Downloader Engine is Active (1.py)'
    })

@app.route('/api/info', methods=['GET'])
def get_video_info():
    """Extracts video formats, metadata, and proxy stream links."""
    video_url = request.args.get('url')
    if not video_url:
        return jsonify({'success': False, 'error': 'Missing YouTube URL'}), 400

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'format': 'best',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            formats = []
            raw_formats = info.get('formats', [])
            
            for f in raw_formats:
                if f.get('url'):
                    filesize = f.get('filesize') or f.get('filesize_approx') or 0
                    vcodec = f.get('vcodec', 'none')
                    acodec = f.get('acodec', 'none')
                    
                    has_video = vcodec != 'none'
                    has_audio = acodec != 'none'

                    if (has_video and has_audio) or (not has_video and has_audio):
                        res = f.get('resolution') or (f"{f.get('height')}p" if f.get('height') else 'Audio')
                        ext = f.get('ext', 'mp4')
                        
                        formats.append({
                            'format_id': f.get('format_id'),
                            'resolution': res,
                            'ext': ext,
                            'filesize_mb': format_bytes(filesize),
                            'has_video': has_video,
                            'has_audio': has_audio,
                            'download_url': f"/api/proxy?stream_url={requests.utils.quote(f['url'])}"
                        })

            formats.reverse()

            return jsonify({
                'success': True,
                'title': info.get('title', 'YouTube Video'),
                'thumbnail': info.get('thumbnail', ''),
                'duration': info.get('duration', 0),
                'channel': info.get('uploader', 'Unknown Channel'),
                'view_count': info.get('view_count', 0),
                'formats': formats[:8]
            })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/proxy', methods=['GET'])
def proxy_stream():
    """Proxies YouTube direct stream through Flask backend to bypass browser CORS."""
    stream_url = request.args.get('stream_url')
    custom_filename = request.args.get('filename', 'download.mp4')

    if not stream_url:
        return "Missing stream URL", 400

    try:
        req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(stream_url, headers=req_headers, stream=True)

        response = Response(
            res.iter_content(chunk_size=1024 * 1024),
            content_type=res.headers.get('Content-Type', 'video/mp4'),
            status=res.status_code
        )

        safe_name = clean_filename(custom_filename)
        response.headers['Content-Disposition'] = f'attachment; filename="{safe_name}"'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    except Exception as e:
        return f"Proxy Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting MediaFetch Engine (1.py) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
