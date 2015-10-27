package mlsp.cs.cmu.edu.dnn.elements;


public class SwitchEdge extends Edge implements Switchable {

  private static final long serialVersionUID = 1338100192118401076L;
  
  private Edge edge;
  private boolean switchOff;
  
  public SwitchEdge(Edge e) {
    this.edge = e;
    this.switchOff = false;
  }
  
  @Override
  public void setSwitchOff(boolean b) {
    switchOff = b;
  }
  
  @Override
  protected void updateWeight() {
    if(!switchOff)
      edge.updateWeight();
  }
  
  @Override
  public void batchUpdate() {
    if(!switchOff)
      edge.batchUpdate();
  }

  @Override
  public void forward() {
    edge.forward();
  }

  @Override
  public void backward() {
    edge.setGradient(edge.getOutgoingElement().getGradient() * edge.derivative());
    updateWeight();
  }

  @Override
  public double derivative() {
    return edge.derivative();
  }

  @Override
  public double getOutput() {
    return edge.getOutput();
  }

  @Override
  public double getGradient() {
    return edge.getGradient();
  }

}
