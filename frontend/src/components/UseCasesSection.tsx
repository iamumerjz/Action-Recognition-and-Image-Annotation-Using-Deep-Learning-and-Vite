import { Card } from "@/components/ui/card";
import { Activity, Shield, Accessibility, Video, Users, TrendingUp } from "lucide-react";

interface UseCase {
  icon: React.ReactNode;
  title: string;
  description: string;
  gradient: string;
}

const useCases: UseCase[] = [
  {
    icon: <Activity className="w-8 h-8" />,
    title: "Sports Analysis",
    description: "Track player movements, analyze techniques, and improve athletic performance with AI-powered action recognition.",
    gradient: "from-soft-blue to-primary/30",
  },
  {
    icon: <Shield className="w-8 h-8" />,
    title: "Security & Surveillance",
    description: "Detect suspicious activities and unusual behaviors in real-time for enhanced security monitoring.",
    gradient: "from-pastel-purple to-secondary/30",
  },
  {
    icon: <Accessibility className="w-8 h-8" />,
    title: "Accessibility Tools",
    description: "Help visually impaired users understand visual content through accurate action descriptions and captions.",
    gradient: "from-mint-green to-accent/30",
  },
  {
    icon: <Video className="w-8 h-8" />,
    title: "Content Moderation",
    description: "Automatically identify and flag inappropriate actions in user-generated video content.",
    gradient: "from-soft-blue to-mint-green",
  },
  {
    icon: <Users className="w-8 h-8" />,
    title: "Healthcare Monitoring",
    description: "Monitor patient movements and activities for rehabilitation tracking and elderly care.",
    gradient: "from-pastel-purple to-soft-blue",
  },
  {
    icon: <TrendingUp className="w-8 h-8" />,
    title: "Retail Analytics",
    description: "Analyze customer behavior patterns and interactions to optimize store layouts and experiences.",
    gradient: "from-mint-green to-pastel-purple",
  },
];

const UseCasesSection = () => {
  return (
    <section id="use-cases" className="py-20 bg-background">
      <div className="container px-4">
        <div className="max-w-4xl mx-auto text-center mb-12 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Real-World Applications
          </h2>
          <p className="text-muted-foreground">
            Discover how action recognition AI is transforming industries
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto stagger-children">
          {useCases.map((useCase, index) => (
            <Card
              key={index}
              variant="elevated"
              className="group overflow-hidden hover:shadow-lg transition-all duration-300"
            >
              <div className="p-6">
                <div
                  className={`w-14 h-14 rounded-xl bg-gradient-to-br ${useCase.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}
                >
                  <div className="text-foreground/80 group-hover:text-foreground transition-colors">
                    {useCase.icon}
                  </div>
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  {useCase.title}
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {useCase.description}
                </p>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default UseCasesSection;
