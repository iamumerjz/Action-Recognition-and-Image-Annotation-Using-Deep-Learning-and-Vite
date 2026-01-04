// frontend/src/pages/Results.tsx
import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, MessageSquare, Trash2, Clock, Loader2, Upload, X, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export interface AnalysisResult {
  id: string;
  imagePreview: string;
  action: string;
  caption: string;
  timestamp: number;
}

const analyzeImage = async (file: File): Promise<{ action: string; caption: string; confidence: number }> => {
  const formData = new FormData();
  formData.append('image', file);

  const response = await fetch(`${API_URL}/api/scan`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to analyze image');
  }

  const data = await response.json();
  return {
    action: data.action,
    caption: data.caption,
    confidence: data.confidence
  };
};

const Results = () => {
  const { toast } = useToast();
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const loadResults = () => {
    const stored = localStorage.getItem("analysisResults");
    if (stored) {
      setResults(JSON.parse(stored));
    }
  };

  useEffect(() => {
    loadResults();
  }, []);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFile(files[0]);
    }
  };

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
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    
    try {
      const imagePreview = preview!;
      
      toast({
        title: "Processing...",
        description: "Analyzing your image with AI",
      });

      const data = await analyzeImage(selectedFile);
      
      const result: AnalysisResult = {
        id: crypto.randomUUID(),
        imagePreview,
        action: data.action,
        caption: data.caption,
        timestamp: Date.now(),
      };
      
      // Save to localStorage
      const existing = JSON.parse(localStorage.getItem("analysisResults") || "[]");
      const updated = [result, ...existing];
      localStorage.setItem("analysisResults", JSON.stringify(updated));
      setResults(updated);
      
      // Clear form
      setSelectedFile(null);
      setPreview(null);
      
      toast({
        title: "Analysis Complete!",
        description: `Detected: ${data.action} (${(data.confidence * 100).toFixed(0)}% confidence)`,
      });
    } catch (error) {
      console.error('Analysis error:', error);
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "There was an error processing your image.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearHistory = () => {
    localStorage.removeItem("analysisResults");
    setResults([]);
    toast({
      title: "History Cleared",
      description: "All analysis results have been removed",
    });
  };

  const deleteResult = (id: string) => {
    const updated = results.filter((r) => r.id !== id);
    localStorage.setItem("analysisResults", JSON.stringify(updated));
    setResults(updated);
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar onTryDemo={() => window.scrollTo({ top: 0, behavior: 'smooth' })} />

      {/* Upload Section */}
      <section className="py-20 bg-muted/30">
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
                    <Upload className="w-10 h-10 text-primary" />
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
                  onClick={handleAnalyze}
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

      {/* Results Section */}
      <main className="py-20">
        <div className="container px-4">
          <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
              <div>
                <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                  Analysis History
                </h2>
                <p className="text-muted-foreground mt-2">
                  View all your analyzed images and their AI-generated captions
                </p>
              </div>

              {results.length > 0 && (
                <Button variant="destructive" onClick={clearHistory}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear History
                </Button>
              )}
            </div>

            {/* Results Grid */}
            {results.length === 0 ? (
              <Card variant="soft" className="text-center py-16">
                <CardContent>
                  <p className="text-muted-foreground text-lg">
                    No analysis results yet. Upload an image above to get started!
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-6">
                {results.map((result, index) => (
                  <Card
                    key={result.id}
                    variant={index === 0 ? "purple" : "soft"}
                    className="overflow-hidden animate-fade-in"
                  >
                    <div className="grid md:grid-cols-[300px_1fr] gap-6 p-6">
                      {/* Image Preview */}
                      <div className="relative">
                        <img
                          src={result.imagePreview}
                          alt="Analyzed image"
                          className="w-full h-48 md:h-full object-cover rounded-xl"
                        />
                        {index === 0 && (
                          <div className="absolute top-2 left-2 px-2 py-1 bg-primary text-primary-foreground text-xs font-medium rounded-full">
                            Latest
                          </div>
                        )}
                      </div>

                      {/* Results */}
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
                          <Clock className="w-3 h-3" />
                          {formatDate(result.timestamp)}
                        </div>

                        <div className="grid sm:grid-cols-2 gap-4 flex-1">
                          {/* Action */}
                          <div className="bg-card rounded-xl p-4 shadow-soft">
                            <div className="flex items-center gap-2 mb-2">
                              <Activity className="w-4 h-4 text-primary" />
                              <span className="text-sm font-medium text-muted-foreground">
                                Recognized Action
                              </span>
                            </div>
                            <p className="text-xl font-bold text-foreground">
                              {result.action}
                            </p>
                          </div>

                          {/* Caption */}
                          <div className="bg-card rounded-xl p-4 shadow-soft">
                            <div className="flex items-center gap-2 mb-2">
                              <MessageSquare className="w-4 h-4 text-secondary" />
                              <span className="text-sm font-medium text-muted-foreground">
                                Generated Caption
                              </span>
                            </div>
                            <p className="text-foreground leading-relaxed">
                              {result.caption}
                            </p>
                          </div>
                        </div>

                        <div className="mt-4 flex justify-end">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => deleteResult(result.id)}
                            className="text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="w-4 h-4 mr-1" />
                            Remove
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Results;