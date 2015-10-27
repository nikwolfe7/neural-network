package mlsp.cs.cmu.edu.dnn.elements;

import java.io.Serializable;

public interface NetworkElement extends Serializable {
  
  /**
   * Forward propagation for this element. 
   * 
   * For weights, just the previous node output 
   * times the weight
   * 
   * For neurons, sum the weight inputs and squish
   * them through your activation function
   */
  public void forward();
  
  /**
   * Backpropagation for this element. 
   * 
   * Should use whichever functions necessary 
   * (usually the derivative), and sets the 
   * computed gradient parameter for a particular
   * training instance
   */
  public void backward();
  
  /**
   * Defines the function to compute the derivative for
   * this element with respect to its output
   * 
   * @return
   */
  public double derivative();
  
  /**
   * Returns the output/activation for this 
   * element
   * 
   * @return
   */
  public double getOutput();
  
  /**
   * Returns the computed gradient for this trainining
   * instance. The difference between this function and
   * {@link #derivative()} is that derivative defines the
   * derivative function, and this returns the computed 
   * gradient for a given training instance
   * 
   * @return
   */
  public double getGradient();
  
}
