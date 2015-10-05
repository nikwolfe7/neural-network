package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.Input;

public class InputLayer extends NetworkElementLayer {

  private Input[] inputs;
  
  public InputLayer(Input... elements) {
    super(elements);
    this.inputs = elements;
  }
  
  public void setInputVector(double[] input) {
    for(int i = 0; i < inputs.length; i++) {
      inputs[i].setInputValue(input[i]);
    }
  }

}
