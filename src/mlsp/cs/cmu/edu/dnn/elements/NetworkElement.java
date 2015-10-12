package mlsp.cs.cmu.edu.dnn.elements;

import java.io.Serializable;

public interface NetworkElement extends Serializable {
  
  public void forward();
  
  public void backward();
  
  public double derivative();

  public double getOutput();
  
  public double getErrorTerm();
  
}
