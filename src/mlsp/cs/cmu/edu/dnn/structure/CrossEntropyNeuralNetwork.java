package mlsp.cs.cmu.edu.dnn.structure;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.util.CostFunction;

public class CrossEntropyNeuralNetwork extends NeuralNetwork {

  private static final long serialVersionUID = -8169090354821138749L;

  public CrossEntropyNeuralNetwork(List<Layer> layers) {
    super(layers);
    /* Create softmax layer on the outside */
    Layer softMaxLayer = new SoftMaxLayer(getLastLayer());
  }
  
  @Override
  protected double getErrorTerm(double[] prediction, double[] truth) {
    return CostFunction.crossEntropy(prediction, truth);
  }
  
}
