package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.structure.InputLayer;
import mlsp.cs.cmu.edu.structure.OutputLayer;

public class SimpleDNNFactory implements DNNFactory {

  public SimpleDNNFactory(int inputDimension, int outputDimension, int... hiddenLayerDimensions) {
    Input[] inputs = new Input[inputDimension];
    for(int i = 0; i < inputs.length; i++)
      inputs[i] = new Input();
    InputLayer inputLayer = new InputLayer(inputs);
    Output[] outputs = new Output[outputDimension];
    for(int i = 0; i < outputs.length; i++)
      outputs[i] = new Output();
    OutputLayer outputLayer = new OutputLayer(outputs);
    
  
  }

  @Override
  public NeuralNetwork getInitializedNeuralNetwork() {
    // TODO Auto-generated method stub
    return null;
  }

}
