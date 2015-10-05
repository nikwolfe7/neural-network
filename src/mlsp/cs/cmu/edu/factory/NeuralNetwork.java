package mlsp.cs.cmu.edu.factory;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.structure.InputLayer;
import mlsp.cs.cmu.edu.structure.NetworkElementLayer;
import mlsp.cs.cmu.edu.structure.OutputLayer;

public class NeuralNetwork {
  
  private List<NetworkElementLayer> hiddenLayers;
  private InputLayer inputLayer;
  private OutputLayer outputLayer;
  
  public NeuralNetwork(InputLayer input, OutputLayer output, NetworkElementLayer... hidden) {
	/* using ArrayList b/c the width varies */
	this.inputLayer = input;
	this.outputLayer = output;
	this.hiddenLayers = new ArrayList<>();
	for(NetworkElementLayer layer : hidden)
		hiddenLayers.add(layer);
  }
  
  public double[] forwardPropagate(double[] input) {
	 inputLayer.setInputVector(input);
	 inputLayer.forwardPropagate();
	 for(NetworkElementLayer layer : hiddenLayers)
		 layer.forwardPropagate();
	 outputLayer.forwardPropagate();
	 return outputLayer.getOutput();
  }
  
  public void backPropagate(double[] truthValue) {
	  outputLayer.setTruthValue(truthValue);
	  outputLayer.backPropagate();
	  for(NetworkElementLayer layer : hiddenLayers)
		  layer.backPropagate();
	  inputLayer.backPropagate();
  }


}
