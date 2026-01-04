// backend/src/routes/scan.ts
import { Router } from 'express';
import multer from 'multer';
import path from 'path';
import { unlink, mkdir } from 'fs/promises';
import { spawn } from 'child_process';
import { v4 as uuidv4 } from 'uuid';

const router = Router();

// Setup upload directory
const UPLOAD_DIR = path.join(process.cwd(), 'uploads');
mkdir(UPLOAD_DIR, { recursive: true }).catch(() => {});

// Configure multer
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename: (req, file, cb) => cb(null, `${uuidv4()}${path.extname(file.originalname)}`)
  }),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = /jpeg|jpg|png|gif|webp/;
    if (allowed.test(path.extname(file.originalname).toLowerCase()) && allowed.test(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Only images allowed'));
    }
  }
});

// Run Python model
function runAnalysis(imagePath: string): Promise<{ action: string; caption: string; confidence: number }> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(process.cwd(), 'dl_model/main.py');
    const pythonProcess = spawn('python', [pythonScript, imagePath]);
    
    let output = '';
    let errorOutput = '';
    
    pythonProcess.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      console.log('Python stdout:', text);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const text = data.toString();
      errorOutput += text;
      console.error('Python stderr:', text);
    });
    
    pythonProcess.on('close', (code) => {
      console.log('Python process closed with code:', code);
      
      if (code !== 0) {
        console.error('Python error output:', errorOutput);
        return reject(new Error(`Python failed with code ${code}: ${errorOutput}`));
      }
      
      const lines = output.split('\n');
      let action = null, confidence = 0, caption = null;
      
      for (const line of lines) {
        if (line.includes('ACTION:')) action = line.split('ACTION:')[1]?.trim();
        if (line.includes('Confidence:')) {
          const match = line.match(/(\d+\.?\d*)%/);
          if (match) confidence = parseFloat(match[1]) / 100;
        }
        if (line.includes('CAPTION:')) caption = line.split('CAPTION:')[1]?.trim();
      }
      
      resolve({ action: action || 'Unknown', caption: caption || '', confidence });
    });
    
    setTimeout(() => { pythonProcess.kill(); reject(new Error('Timeout')); }, 60000);
  });
}

// POST /api/scan - Upload and analyze image
router.post('/', upload.single('image'), async (req, res) => {
  console.log('Received scan request');
  
  if (!req.file) {
    console.log('Error: No image provided');
    return res.status(400).json({ error: 'No image provided' });
  }
  
  console.log('File received:', req.file.filename, req.file.mimetype, req.file.size);
  
  try {
    console.log('Starting analysis...');
    const result = await runAnalysis(req.file.path);
    console.log('Analysis result:', result);
    
    await unlink(req.file.path); // Cleanup
    console.log('File cleaned up');
    
    res.json({
      action: result.action,
      caption: result.caption,
      confidence: result.confidence
    });
  } catch (error) {
    console.error('Analysis error:', error);
    if (req.file) await unlink(req.file.path).catch(() => {});
    res.status(500).json({ error: error instanceof Error ? error.message : 'Analysis failed' });
  }
});

export default router;