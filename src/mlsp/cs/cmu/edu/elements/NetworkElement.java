package mlsp.cs.cmu.edu.elements;

public interface NetworkElement {
  
  public void forward();
  
  public void backward();
  
  public double derivative();

  public double getOutput();
  
  public double getErrorTerm();
  
}
