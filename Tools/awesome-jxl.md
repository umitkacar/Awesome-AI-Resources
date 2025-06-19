# 🖼️ Awesome JPEG XL (JXL) Resources

## Overview
A comprehensive collection of resources, tools, and information about JPEG XL (JXL) - the next-generation image format designed to replace JPEG, PNG, and GIF.

## 🎯 What is JPEG XL?

### Key Features
- **Superior Compression**: 20-60% smaller than JPEG at same quality
- **Lossless Compression**: Better than PNG compression ratios
- **Progressive Decoding**: Images appear gradually while loading
- **Wide Gamut**: Supports HDR and wide color gamuts
- **Animation Support**: Can replace animated GIFs efficiently
- **Royalty-Free**: Open standard with no licensing fees
- **Backwards Compatible**: Can losslessly transcode existing JPEGs

### Technical Specifications
```
Format Name: JPEG XL
File Extension: .jxl
MIME Type: image/jxl
Magic Number: 0xFF 0x0A (bare codestream) or "JXL " (container)
Max Dimensions: 1,073,741,823 × 1,073,741,823 pixels
Color Depth: Up to 32 bits per channel
Channels: Up to 4,096 channels
Animation: Supported with custom frame rates
```

## 🛠️ Tools and Libraries

### Official Reference Implementation
```bash
# Build libjxl from source
git clone https://github.com/libjxl/libjxl.git
cd libjxl
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF ..
make -j$(nproc)

# Command-line tools
cjxl input.png output.jxl -q 90        # Encode
djxl input.jxl output.png              # Decode
jxlinfo image.jxl                      # Get image info
```

### Encoding Options
```bash
# High quality lossy compression
cjxl input.png output.jxl --quality 90 --effort 7

# Lossless compression
cjxl input.png output.jxl --lossless --effort 9

# From JPEG (lossless transcoding)
cjxl input.jpg output.jxl --lossless_jpeg

# Animation from multiple frames
cjxl frame*.png animation.jxl --quality 85

# With custom settings
cjxl input.png output.jxl \
  --quality 92 \
  --effort 8 \
  --brotli_effort 11 \
  --progressive \
  --photon_noise_iso 3200
```

## 💻 Programming Libraries

### Python - imagecodecs
```python
import imagecodecs
import numpy as np
from PIL import Image

# Encode numpy array to JXL
def encode_jxl(image_array, quality=90):
    """Encode numpy array to JXL bytes"""
    jxl_bytes = imagecodecs.jpegxl_encode(
        image_array,
        lossless=False,
        quality=quality,
        effort=7
    )
    return jxl_bytes

# Decode JXL to numpy array
def decode_jxl(jxl_bytes):
    """Decode JXL bytes to numpy array"""
    return imagecodecs.jpegxl_decode(jxl_bytes)

# Convert PIL Image to JXL
def pil_to_jxl(pil_image, output_path, quality=90):
    """Convert PIL Image to JXL file"""
    # Convert to numpy array
    img_array = np.array(pil_image)
    
    # Encode to JXL
    jxl_bytes = encode_jxl(img_array, quality)
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(jxl_bytes)

# Example usage
image = Image.open('photo.jpg')
pil_to_jxl(image, 'photo.jxl', quality=92)
```

### C++ Integration
```cpp
#include <jxl/encode.h>
#include <jxl/decode.h>
#include <vector>

class JXLProcessor {
private:
    JxlEncoder* encoder;
    JxlDecoder* decoder;
    
public:
    JXLProcessor() {
        encoder = JxlEncoderCreate(nullptr);
        decoder = JxlDecoderCreate(nullptr);
    }
    
    std::vector<uint8_t> encode(const uint8_t* pixels, 
                                int width, int height, 
                                int channels, float quality) {
        // Basic info
        JxlBasicInfo basic_info;
        JxlEncoderInitBasicInfo(&basic_info);
        basic_info.xsize = width;
        basic_info.ysize = height;
        basic_info.bits_per_sample = 8;
        basic_info.num_color_channels = 3;
        
        if (channels == 4) {
            basic_info.alpha_bits = 8;
            basic_info.num_extra_channels = 1;
        }
        
        JxlEncoderSetBasicInfo(encoder, &basic_info);
        
        // Color encoding
        JxlColorEncoding color_encoding = {};
        JxlColorEncodingSetToSRGB(&color_encoding, 
                                  JXL_FALSE);
        JxlEncoderSetColorEncoding(encoder, &color_encoding);
        
        // Frame settings
        JxlEncoderFrameSettings* frame_settings = 
            JxlEncoderFrameSettingsCreate(encoder, nullptr);
        
        // Set quality (distance parameter)
        float distance = quality >= 100 ? 0.0f : 
                        (100.0f - quality) / 10.0f;
        JxlEncoderSetFrameDistance(frame_settings, distance);
        
        // Add image frame
        JxlPixelFormat pixel_format = {channels, JXL_TYPE_UINT8, 
                                      JXL_NATIVE_ENDIAN, 0};
        
        JxlEncoderAddImageFrame(frame_settings, &pixel_format, 
                               pixels, width * height * channels);
        
        // Encode
        std::vector<uint8_t> compressed;
        compressed.resize(width * height * channels); // Initial size
        uint8_t* next_out = compressed.data();
        size_t avail_out = compressed.size();
        
        JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
        while (status == JXL_ENC_NEED_MORE_OUTPUT) {
            status = JxlEncoderProcessOutput(encoder, &next_out, &avail_out);
            if (status == JXL_ENC_NEED_MORE_OUTPUT) {
                size_t offset = next_out - compressed.data();
                compressed.resize(compressed.size() * 2);
                next_out = compressed.data() + offset;
                avail_out = compressed.size() - offset;
            }
        }
        
        compressed.resize(next_out - compressed.data());
        return compressed;
    }
    
    ~JXLProcessor() {
        JxlEncoderDestroy(encoder);
        JxlDecoderDestroy(decoder);
    }
};
```

### JavaScript/WebAssembly
```javascript
// Using jxl-wasm
import { encode, decode } from '@jsquash/jxl';

async function convertToJXL(imageFile) {
    // Read file as ArrayBuffer
    const arrayBuffer = await imageFile.arrayBuffer();
    
    // Decode source image (PNG/JPEG)
    const image = new Image();
    const blob = new Blob([arrayBuffer]);
    const url = URL.createObjectURL(blob);
    
    return new Promise((resolve, reject) => {
        image.onload = async () => {
            // Create canvas
            const canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            
            // Get image data
            const imageData = ctx.getImageData(0, 0, 
                                             canvas.width, 
                                             canvas.height);
            
            // Encode to JXL
            const jxlBuffer = await encode(imageData, {
                quality: 90,
                effort: 7
            });
            
            resolve(new Blob([jxlBuffer], { type: 'image/jxl' }));
            URL.revokeObjectURL(url);
        };
        
        image.onerror = reject;
        image.src = url;
    });
}

// Display JXL in browser (with polyfill)
async function displayJXL(jxlBlob, imgElement) {
    const arrayBuffer = await jxlBlob.arrayBuffer();
    const imageData = await decode(new Uint8Array(arrayBuffer));
    
    // Create canvas and draw decoded image
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    
    // Convert to data URL
    imgElement.src = canvas.toDataURL();
}
```

## 🌐 Browser Support

### Current Status (2024)
```javascript
// Feature detection
function supportsJXL() {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 1;
    const ctx = canvas.getContext('2d');
    
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(true);
        img.onerror = () => resolve(false);
        
        // 1x1 JXL image
        img.src = 'data:image/jxl;base64,/woAELASCAgQAFwASxLFgkWAHL0xqnCBCV0qDp901Te/5QM=';
    });
}

// Progressive enhancement
async function loadImage(src) {
    const img = document.createElement('img');
    
    if (await supportsJXL()) {
        img.src = src.replace(/\.(jpg|png)$/i, '.jxl');
    } else {
        img.src = src;
    }
    
    return img;
}
```

### Polyfill Solution
```html
<!-- Include JXL polyfill -->
<script src="https://unpkg.com/jxl-js@1.0.0/jxl.js"></script>

<script>
// Automatic polyfill for all JXL images
if (!supportsJXL()) {
    document.addEventListener('DOMContentLoaded', () => {
        const images = document.querySelectorAll('img[src$=".jxl"]');
        images.forEach(img => {
            fetch(img.src)
                .then(res => res.arrayBuffer())
                .then(buffer => {
                    const decoded = JXL.decode(new Uint8Array(buffer));
                    const blob = new Blob([decoded], { type: 'image/png' });
                    img.src = URL.createObjectURL(blob);
                });
        });
    });
}
</script>
```

## 📊 Compression Comparisons

### Benchmark Results
```python
import os
import subprocess
import time
from PIL import Image
import pandas as pd

def benchmark_compression(input_path, quality_levels=[70, 80, 90, 95]):
    results = []
    
    for quality in quality_levels:
        # JPEG
        jpeg_path = f'temp_q{quality}.jpg'
        img = Image.open(input_path)
        img.save(jpeg_path, 'JPEG', quality=quality, optimize=True)
        jpeg_size = os.path.getsize(jpeg_path)
        
        # WebP
        webp_path = f'temp_q{quality}.webp'
        img.save(webp_path, 'WebP', quality=quality, method=6)
        webp_size = os.path.getsize(webp_path)
        
        # AVIF
        avif_path = f'temp_q{quality}.avif'
        subprocess.run(['avifenc', input_path, avif_path, 
                       '-q', str(quality)], capture_output=True)
        avif_size = os.path.getsize(avif_path) if os.path.exists(avif_path) else 0
        
        # JXL
        jxl_path = f'temp_q{quality}.jxl'
        start_time = time.time()
        subprocess.run(['cjxl', input_path, jxl_path, 
                       '-q', str(quality), '-e', '7'], capture_output=True)
        encode_time = time.time() - start_time
        jxl_size = os.path.getsize(jxl_path)
        
        results.append({
            'Quality': quality,
            'JPEG (KB)': jpeg_size / 1024,
            'WebP (KB)': webp_size / 1024,
            'AVIF (KB)': avif_size / 1024,
            'JXL (KB)': jxl_size / 1024,
            'JXL vs JPEG': f'{(1 - jxl_size/jpeg_size) * 100:.1f}%',
            'Encode Time': f'{encode_time:.2f}s'
        })
        
        # Cleanup
        for path in [jpeg_path, webp_path, avif_path, jxl_path]:
            if os.path.exists(path):
                os.remove(path)
    
    return pd.DataFrame(results)
```

## 🔧 Server Configuration

### Nginx
```nginx
# Add MIME type
types {
    image/jxl jxl;
}

# Enable gzip compression for JXL
gzip_types image/jxl;

# Content negotiation
location ~* \.(jpg|jpeg|png)$ {
    add_header Vary Accept;
    
    if ($http_accept ~* "image/jxl") {
        rewrite ^(.*)\.jpg$ $1.jxl last;
        rewrite ^(.*)\.jpeg$ $1.jxl last;
        rewrite ^(.*)\.png$ $1.jxl last;
    }
}
```

### Apache
```apache
# .htaccess configuration
AddType image/jxl .jxl

# Content negotiation
<IfModule mod_rewrite.c>
    RewriteEngine On
    
    # Check if browser accepts JXL
    RewriteCond %{HTTP_ACCEPT} image/jxl
    
    # Check if JXL version exists
    RewriteCond %{REQUEST_FILENAME}.jxl -f
    
    # Serve JXL instead
    RewriteRule ^(.+)\.(jpe?g|png)$ $1.jxl [T=image/jxl,L]
</IfModule>
```

### CDN Configuration
```javascript
// Cloudflare Worker for JXL delivery
addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
    const url = new URL(request.url);
    const accept = request.headers.get('Accept') || '';
    
    // Check if browser supports JXL
    if (accept.includes('image/jxl') && 
        /\.(jpg|jpeg|png)$/i.test(url.pathname)) {
        
        // Try to fetch JXL version
        const jxlUrl = url.pathname.replace(/\.(jpg|jpeg|png)$/i, '.jxl');
        const jxlResponse = await fetch(url.origin + jxlUrl);
        
        if (jxlResponse.ok) {
            return new Response(jxlResponse.body, {
                headers: {
                    'Content-Type': 'image/jxl',
                    'Cache-Control': 'public, max-age=31536000',
                    'Vary': 'Accept'
                }
            });
        }
    }
    
    // Fallback to original
    return fetch(request);
}
```

## 🎨 Use Cases

### Photography
- **RAW Alternative**: Lossless compression for archival
- **Web Galleries**: Faster loading with better quality
- **HDR Images**: Native HDR support
- **Batch Processing**: Efficient storage for large collections

### Web Development
- **Progressive Loading**: Better user experience
- **Responsive Images**: Single file for all resolutions
- **Animation**: Replace heavy GIFs
- **Bandwidth Savings**: Reduce CDN costs

### Machine Learning
```python
# Using JXL for ML datasets
import imagecodecs
import numpy as np
from pathlib import Path

class JXLDataset:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.images = list(self.root_dir.glob('*.jxl'))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load JXL image
        with open(self.images[idx], 'rb') as f:
            jxl_data = f.read()
        
        # Decode to numpy array
        img_array = imagecodecs.jpegxl_decode(jxl_data)
        
        # Normalize for ML (0-1 range)
        if img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def save_dataset_as_jxl(self, images, labels, output_dir):
        """Convert existing dataset to JXL format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for i, (img, label) in enumerate(zip(images, labels)):
            # Encode to JXL
            jxl_data = imagecodecs.jpegxl_encode(
                img, 
                lossless=False,
                quality=95,
                effort=7
            )
            
            # Save with label in filename
            output_path = output_dir / f'{i:06d}_label{label}.jxl'
            with open(output_path, 'wb') as f:
                f.write(jxl_data)
```

## 🔗 Resources

### Official Resources
- **Specification**: gitlab.com/wg1/jpeg-xl
- **Reference Implementation**: github.com/libjxl/libjxl
- **Test Images**: github.com/libjxl/testdata
- **Conformance**: github.com/libjxl/conformance

### Community
- **Discord**: JPEG XL Discord Server
- **Reddit**: r/jpegxl
- **Blog**: jpegxl.info
- **Demos**: jpegxl.io

### Tools & Viewers
- **ImageMagick**: Supports JXL with delegates
- **GIMP**: Plugin available
- **IrfanView**: Native support
- **XnView**: JXL plugin

### Performance Tools
- **Benchmark Suite**: github.com/libjxl/benchmark
- **Quality Metrics**: SSIMULACRA2, BUTTERAUGLI
- **Comparison Tool**: github.com/google/butteraugli

---

*The future of image compression is here with JPEG XL* 🖼️🚀