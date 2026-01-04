import { Card } from "@/components/ui/card";
import { Upload, Cpu, Brain, MessageSquare } from "lucide-react";

const steps = [
  {
    step: 1,
    title: "Upload Image",
    description: "Select or drag & drop an image containing human activity",
    icon: <Upload className="w-8 h-8" />,
    color: "bg-soft-blue text-primary",
  },
  {
    step: 2,
    title: "CNN Extracts Features",
    description: "Convolutional Neural Network analyzes visual patterns and objects",
    icon: <Cpu className="w-8 h-8" />,
    color: "bg-pastel-purple text-secondary",
  },
  {
    step: 3,
    title: "LSTM Processes Sequence",
    description: "Long Short-Term Memory network understands temporal relationships",
    icon: <Brain className="w-8 h-8" />,
    color: "bg-mint-green text-accent",
  },
  {
    step: 4,
    title: "Results Generated",
    description: "AI outputs recognized action and descriptive caption",
    icon: <MessageSquare className="w-8 h-8" />,
    color: "bg-soft-blue text-primary",
  },
];

const HowItWorksSection = () => {
  return (
    <section className="py-20 bg-muted/30">
      <div className="container px-4">
        <div className="max-w-4xl mx-auto text-center mb-16 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            How It Works
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Our deep learning pipeline processes your image through multiple stages 
            to understand and describe human actions
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-4 gap-6 stagger-children">
            {steps.map((item, index) => (
              <div key={item.step} className="relative">
                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-12 left-1/2 w-full h-0.5 bg-gradient-to-r from-border via-primary/30 to-border z-0" />
                )}
                
                <Card variant="glass" className="relative z-10 p-6 text-center h-full">
                  {/* Step number */}
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-xs font-bold">
                    {item.step}
                  </div>
                  
                  {/* Icon */}
                  <div className={`w-16 h-16 ${item.color} rounded-2xl flex items-center justify-center mx-auto mb-4`}>
                    {item.icon}
                  </div>
                  
                  {/* Content */}
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    {item.title}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {item.description}
                  </p>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorksSection;
