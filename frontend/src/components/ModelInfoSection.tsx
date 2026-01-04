import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Cpu, Brain, Database, Layers } from "lucide-react";

const ModelInfoSection = () => {
  return (
    <section className="py-20 bg-background">
      <div className="container px-4">
        <div className="max-w-4xl mx-auto text-center mb-12 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Model Architecture
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Built with state-of-the-art deep learning techniques for accurate 
            action recognition and natural language generation
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto stagger-children">
          {/* CNN Card */}
          <Card variant="soft" className="overflow-hidden">
            <CardHeader>
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 bg-primary/20 rounded-2xl flex items-center justify-center">
                  <Cpu className="w-7 h-7 text-primary" />
                </div>
                <div>
                  <CardTitle className="text-xl">CNN - Feature Extraction</CardTitle>
                  <p className="text-sm text-muted-foreground">Convolutional Neural Network</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                The CNN backbone extracts hierarchical visual features from input images, 
                identifying objects, textures, and spatial relationships crucial for 
                understanding human activities.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-soft-blue rounded-full text-xs font-medium text-primary">
                  VGG/ResNet Base
                </span>
                <span className="px-3 py-1 bg-soft-blue rounded-full text-xs font-medium text-primary">
                  Feature Maps
                </span>
                <span className="px-3 py-1 bg-soft-blue rounded-full text-xs font-medium text-primary">
                  Spatial Analysis
                </span>
              </div>
            </CardContent>
          </Card>

          {/* LSTM Card */}
          <Card variant="purple" className="overflow-hidden">
            <CardHeader>
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 bg-secondary/20 rounded-2xl flex items-center justify-center">
                  <Brain className="w-7 h-7 text-secondary" />
                </div>
                <div>
                  <CardTitle className="text-xl">LSTM - Sequence Processing</CardTitle>
                  <p className="text-sm text-muted-foreground">Long Short-Term Memory</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                The LSTM network processes extracted features sequentially, learning 
                temporal dependencies and context to classify actions and generate 
                coherent descriptive captions.
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-pastel-purple rounded-full text-xs font-medium text-secondary">
                  Memory Gates
                </span>
                <span className="px-3 py-1 bg-pastel-purple rounded-full text-xs font-medium text-secondary">
                  Sequence Learning
                </span>
                <span className="px-3 py-1 bg-pastel-purple rounded-full text-xs font-medium text-secondary">
                  Caption Generation
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Additional Info */}
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto mt-8">
          <Card variant="mint" className="p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-accent/20 rounded-xl flex items-center justify-center flex-shrink-0">
                <Database className="w-6 h-6 text-accent" />
              </div>
              <div>
                <h4 className="font-semibold text-foreground mb-1">Training Dataset</h4>
                <p className="text-sm text-muted-foreground">
                  Trained on diverse human activity datasets including UCF-101, 
                  HMDB51, and custom labeled action sequences for robust recognition.
                </p>
              </div>
            </div>
          </Card>

          <Card variant="glass" className="p-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center flex-shrink-0">
                <Layers className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h4 className="font-semibold text-foreground mb-1">Architecture Details</h4>
                <p className="text-sm text-muted-foreground">
                  End-to-end trainable pipeline with attention mechanisms for 
                  improved focus on relevant image regions during caption generation.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default ModelInfoSection;
