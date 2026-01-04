// frontend/src/pages/Index.tsx
import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import ImageUploadSection from "@/components/ImageUploadSection";
import UseCasesSection from "@/components/UseCasesSection";
import HowItWorksSection from "@/components/HowItWorksSection";
import ModelInfoSection from "@/components/ModelInfoSection";
import Footer from "@/components/Footer";
import type { AnalysisResult } from "./Results";

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

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

const saveResult = (result: AnalysisResult) => {
  const stored = localStorage.getItem("analysisResults");
  const existing: AnalysisResult[] = stored ? JSON.parse(stored) : [];
  const updated = [result, ...existing];
  localStorage.setItem("analysisResults", JSON.stringify(updated));
};

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
  });
};

const Index = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const uploadSectionRef = useRef<HTMLDivElement>(null);

  const scrollToUpload = () => {
    document.getElementById("upload-section")?.scrollIntoView({ behavior: "smooth" });
  };

  const handleAnalyze = async (file: File) => {
    setIsLoading(true);
    
    try {
      const imagePreview = await fileToBase64(file);
      
      toast({
        title: "Processing...",
        description: "Analyzing your image with AI",
      });

      console.log('Sending request to:', `${API_URL}/api/scan`);
      console.log('File:', file.name, file.type, file.size);

      const data = await analyzeImage(file);
      
      console.log('Received data:', data);
      
      const result: AnalysisResult = {
        id: crypto.randomUUID(),
        imagePreview,
        action: data.action,
        caption: data.caption,
        timestamp: Date.now(),
      };
      
      saveResult(result);
      
      toast({
        title: "Analysis Complete!",
        description: `Detected: ${data.action} (${(data.confidence * 100).toFixed(0)}% confidence)`,
      });
      
      navigate("/results");
    } catch (error) {
      console.error('Full error:', error);
      toast({
        title: "Analysis Failed",
        description: error instanceof Error ? error.message : "There was an error processing your image.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar onTryDemo={scrollToUpload} />
      
      <main>
        <HeroSection onTryDemo={scrollToUpload} />
        
        <div ref={uploadSectionRef}>
          <ImageUploadSection onAnalyze={handleAnalyze} isLoading={isLoading} />
        </div>
        
        <div id="use-cases">
          <UseCasesSection />
        </div>

        <div id="applications">
          <UseCasesSection />
        </div>
        
        <div id="how-it-works">
          <HowItWorksSection />
        </div>
        
        <div id="model">
          <ModelInfoSection />
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;