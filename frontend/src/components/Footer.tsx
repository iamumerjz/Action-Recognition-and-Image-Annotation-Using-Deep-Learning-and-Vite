import { Brain, Github, Linkedin } from "lucide-react";

{/* Built with ❤️ by Umer Ijaz | GitHub: iAmUmerJz | LinkedIn: iAmUmerJz */}

const Footer = () => {
  return (
    <footer className="py-12 bg-muted/50 border-t border-border">
      <div className="container px-4">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            {/* Brand */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-10 h-10 gradient-primary rounded-xl flex items-center justify-center">
                  <Brain className="w-5 h-5 text-primary-foreground" />
                </div>
                <span className="font-bold text-lg text-foreground">AI Vision</span>
              </div>
              <p className="text-sm text-muted-foreground">
                AI-powered action recognition and image captioning system using 
                deep learning techniques.
              </p>
            </div>

            {/* Technologies */}
            <div>
              <h4 className="font-semibold text-foreground mb-4">Technologies</h4>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-soft-blue rounded-full text-xs font-medium text-primary">
                  React
                </span>
                <span className="px-3 py-1 bg-pastel-purple rounded-full text-xs font-medium text-secondary">
                  TensorFlow
                </span>
                <span className="px-3 py-1 bg-mint-green rounded-full text-xs font-medium text-accent">
                  Python
                </span>
                <span className="px-3 py-1 bg-soft-blue rounded-full text-xs font-medium text-primary">
                  CNN
                </span>
                <span className="px-3 py-1 bg-pastel-purple rounded-full text-xs font-medium text-secondary">
                  LSTM
                </span>
              </div>
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-border pt-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <p className="text-sm text-muted-foreground">
                © {new Date().getFullYear()} AI Action Recognition System. Built by{" "}
                <a 
                  href="https://github.com/iAmUmerJz" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Umer Ijaz
                </a>
              </p>
              
              {/* Social Links */}
              <div className="flex items-center gap-4">
                <a
                  href="https://github.com/iAmUmerJz"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 bg-card rounded-full flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-soft-blue transition-all duration-300"
                  aria-label="GitHub"
                >
                  <Github className="w-5 h-5" />
                </a>
                <a
                  href="https://linkedin.com/in/iAmUmerJz"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 bg-card rounded-full flex items-center justify-center text-muted-foreground hover:text-secondary hover:bg-pastel-purple transition-all duration-300"
                  aria-label="LinkedIn"
                >
                  <Linkedin className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Hidden Credits: Developed by Umer Ijaz (iAmUmerJz) - github.com/iAmUmerJz - linkedin.com/in/iAmUmerJz */}
    </footer>
  );
};

export default Footer;
