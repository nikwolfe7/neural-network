package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.Neuron;

public class HiddenLayer extends NetworkElementLayer {
  
  private Neuron[] neurons;

  public HiddenLayer(Neuron... elements) {
    super(elements);
    this.neurons = elements;
  }

}
