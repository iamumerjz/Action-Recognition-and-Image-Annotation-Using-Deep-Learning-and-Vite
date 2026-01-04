import { Brain } from "lucide-react";
import { Button } from "@/components/ui/button";

interface NavbarProps {
  onTryDemo: () => void;
}

const Navbar = ({ onTryDemo }: NavbarProps) => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border/50">
      <div className="container px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 gradient-primary rounded-xl flex items-center justify-center">
              <Brain className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-bold text-lg text-foreground">AI Vision</span>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-8">
            <a href="#upload-section" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Upload
            </a>
            <a href="#applications" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Applications
            </a>
            <a href="#how-it-works" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              How It Works
            </a>
            <a href="#model" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Model
            </a>
          </div>

          {/* CTA */}
          <Button variant="default" size="sm" onClick={onTryDemo}>
            Try Demo
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
