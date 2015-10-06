package mlsp.cs.cmu.edu.factory;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.elements.*;
import mlsp.cs.cmu.edu.structure.*;

public class FeedForwardDNNFactory implements DNNFactory {

	private NeuralNetwork network;
	private NetworkElementAbstractFactory factory;
	
	public FeedForwardDNNFactory(int inputDimension, int outputDimension, int... hiddenLayerDimenions) {
	  this.factory = new SigmoidNetworkAbstractFactoryImpl();
		List<Layer> layers = new ArrayList<>();
		
		/* Input layer */
		Input[] inputs  = new Input[inputDimension];
		for(int i = 0; i < inputs.length; i++) {
		  inputs[i] = factory.getNewInput();
		}
		Layer inputLayer = new NetworkElementLayer(inputs);
		layers.add(inputLayer);
		
		/* Output layer */
		Output[] outputs = new Output[outputDimension];
		for(int i = 0 ; i < outputs.length; i++) {
		  outputs[i] = factory.getNewOutput();
		}
		Layer outputLayer = new NetworkElementLayer(outputs);
		
		/* Hidden Layers */
		for(int i = 0; i < hiddenLayerDimenions.length; i++) {
		  /* previous layer */
		  Layer prev = layers.get(layers.size()-1); 
		  int dim = hiddenLayerDimenions[i];
		  Neuron[] hl = new Neuron[dim];
		  for(int j = 0; j < hl.length; j++) {
		    hl[j] = factory.getNewNeuron();
		  }
		  Layer hiddenLayer = new NetworkElementLayer(hl);
		  /* Add the Bias */
		  Bias b = factory.getNewBias();
		  hiddenLayer.addNetworkElements(b);
		  /* Connect the layers */
		  Layer weightMatrix = connect(prev, hiddenLayer);
		  /* Load them into the network stack */
		  layers.add(weightMatrix);
		  layers.add(hiddenLayer);
		}
	}
	
	
	private Layer connect(Layer prev, Layer hiddenLayer) {
    int rows, cols;
    rows = hiddenLayer.size();
    cols = prev.size();
    Edge[] weightMatrix = new Edge[rows * cols];
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        
      }
    }
  }
	
  private void attachElements(Neuron in, Edge w, Neuron out) {
    w.setIncomingElement(in);
    w.setOutgoingElement(out);
    in.addOutgoingElement(w);
    out.addIncomingElement(w);
  }


  @Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		return network;
	}
	
	

}
