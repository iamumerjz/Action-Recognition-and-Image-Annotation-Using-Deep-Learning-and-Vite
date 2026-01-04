// frontend/src/components/ImageUploadSection.tsx
import { useState, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Upload, Image as ImageIcon, X, Loader2, Sparkles } from "lucide-react";

interface ImageUploadSectionProps {
  onAnalyze: (file: File) => void;
  isLoading: boolean;
}

const ImageUploadSection = ({ onAnalyze, isLoading }: ImageUploadSectionProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith("image/")) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const handleAnalyzeClick = () => {
    if (selectedFile) {
      onAnalyze(selectedFile);
    }
  };

  return (
    <section id="upload-section" className="py-20 bg-muted/30">
      <div className="container px-4">
        <div className="max-w-2xl mx-auto text-center mb-12 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Upload Your Image
          </h2>
          <p className="text-muted-foreground">
            Drag and drop an image or click to browse. Our AI will analyze human actions and generate captions.
          </p>
        </div>

        <Card variant="glass" className="max-w-2xl mx-auto p-8 animate-scale-in">
          <div
            className={`relative border-2 border-dashed rounded-xl transition-all duration-300 ${
              dragActive
                ? "border-primary bg-primary/5 scale-105"
                : preview
                ? "border-accent bg-accent/5"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              disabled={isLoading}
            />

            {preview ? (
              <div className="relative p-4">
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 z-20 p-2 bg-destructive/90 hover:bg-destructive text-destructive-foreground rounded-full transition-colors"
                  disabled={isLoading}
                >
                  <X className="w-4 h-4" />
                </button>
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-80 mx-auto rounded-lg shadow-card object-contain"
                />
                <p className="text-center text-sm text-muted-foreground mt-4">
                  {selectedFile?.name}
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 px-4">
                <div className="w-20 h-20 bg-soft-blue rounded-full flex items-center justify-center mb-6">
                  {dragActive ? (
                    <ImageIcon className="w-10 h-10 text-primary animate-pulse" />
                  ) : (
                    <Upload className="w-10 h-10 text-primary" />
                  )}
                </div>
                <p className="text-lg font-medium text-foreground mb-2">
                  {dragActive ? "Drop your image here" : "Drag & drop your image"}
                </p>
                <p className="text-sm text-muted-foreground">
                  or click to browse from your device
                </p>
                <p className="text-xs text-muted-foreground mt-4">
                  Supports: JPG, PNG, GIF, WebP
                </p>
              </div>
            )}
          </div>

          {preview && (
            <div className="mt-6 flex justify-center animate-fade-in">
              <Button
                variant="hero"
                size="lg"
                onClick={handleAnalyzeClick}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Analyze Image
                  </>
                )}
              </Button>
            </div>
          )}
        </Card>
      </div>
    </section>
  );
};

export default ImageUploadSection;
