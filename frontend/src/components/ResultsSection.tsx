// frontend/src/components/ResultsSection.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, MessageSquare, Sparkles } from "lucide-react";

interface ResultsSectionProps {
  action: string | null;
  caption: string | null;
  isVisible: boolean;
}

const ResultsSection = ({ action, caption, isVisible }: ResultsSectionProps) => {
  if (!isVisible) return null;

  return (
    <section className="py-20 bg-gradient-to-b from-muted/30 to-background">
      <div className="container px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12 animate-fade-in-up">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent/20 rounded-full mb-4">
              <Sparkles className="w-4 h-4 text-accent" />
              <span className="text-sm font-medium text-accent">Analysis Complete</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              AI Results
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6 stagger-children">
            {/* Action Recognition Card */}
            <Card variant="soft" className="overflow-hidden">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-primary/20 rounded-xl flex items-center justify-center">
                    <Activity className="w-6 h-6 text-primary" />
                  </div>
                  <CardTitle className="text-xl">Recognized Action</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="bg-card rounded-xl p-6 shadow-soft">
                  <p className="text-2xl md:text-3xl font-bold text-foreground text-center">
                    {action || "No action detected"}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Image Caption Card */}
            <Card variant="purple" className="overflow-hidden">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-secondary/20 rounded-xl flex items-center justify-center">
                    <MessageSquare className="w-6 h-6 text-secondary" />
                  </div>
                  <CardTitle className="text-xl">Generated Caption</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="bg-card rounded-xl p-6 shadow-soft">
                  <p className="text-lg md:text-xl text-foreground text-center leading-relaxed">
                    {caption || "No caption generated"}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ResultsSection;
