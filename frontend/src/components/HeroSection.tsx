import { Button } from "@/components/ui/button";
import { Brain, Sparkles } from "lucide-react";
import heroBackground from "@/assets/hero-background.png";

interface HeroSectionProps {
  onTryDemo: () => void;
}

const HeroSection = ({ onTryDemo }: HeroSectionProps) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${heroBackground})` }}
      />
      
      {/* Overlay gradient for better text readability */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/70 via-background/50 to-background" />
      
      {/* Floating decorations */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-soft-blue rounded-full opacity-60 animate-float blur-xl" />
      <div className="absolute top-40 right-20 w-32 h-32 bg-pastel-purple rounded-full opacity-50 animate-float blur-xl" style={{ animationDelay: "1s" }} />
      <div className="absolute bottom-40 left-1/4 w-24 h-24 bg-mint-green rounded-full opacity-50 animate-float blur-xl" style={{ animationDelay: "2s" }} />
      
      <div className="container relative z-10 px-4 py-20">
        <div className="max-w-4xl mx-auto text-center stagger-children">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-card/80 backdrop-blur-sm rounded-full shadow-soft mb-8">
            <Brain className="w-5 h-5 text-primary" />
            <span className="text-sm font-medium text-foreground">Deep Learning Powered</span>
            <Sparkles className="w-4 h-4 text-secondary" />
          </div>
          
          {/* Main Headline */}
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground mb-6 leading-tight">
            AI-Powered{" "}
            <span className="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
              Action Recognition
            </span>
            <br />
            & Image Captioning
          </h1>
          
          {/* Subheading */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
            Upload an image and let deep learning understand human actions and generate 
            intelligent captions. Powered by CNN & LSTM neural networks.
          </p>
          
          {/* CTA Button */}
          <Button
            variant="hero"
            size="lg"
            onClick={onTryDemo}
            className="group"
          >
            <Sparkles className="w-5 h-5 group-hover:animate-pulse" />
            Try Demo
          </Button>
          
          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 mt-16 max-w-lg mx-auto">
            <div className="text-center">
              <div className="text-3xl font-bold text-foreground">CNN</div>
              <div className="text-sm text-muted-foreground">Feature Extraction</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-foreground">LSTM</div>
              <div className="text-sm text-muted-foreground">Sequence Analysis</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-foreground">AI</div>
              <div className="text-sm text-muted-foreground">Smart Captions</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-muted-foreground/30 rounded-full flex justify-center pt-2">
          <div className="w-1.5 h-3 bg-muted-foreground/50 rounded-full animate-pulse" />
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
