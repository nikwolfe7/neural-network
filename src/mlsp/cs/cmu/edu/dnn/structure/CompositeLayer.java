package mlsp.cs.cmu.edu.dnn.structure;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;

public class CompositeLayer implements Layer {

  private static final long serialVersionUID = 6555068509293432784L;
  
  private List<Layer> subLayers;
  private int totalSize;
  
  public CompositeLayer(Layer... layers) {
    this.totalSize = 0;
    this.subLayers = new ArrayList<Layer>();
    for(Layer l : layers) {
      subLayers.add(l);
      totalSize += l.size();
    }
  }
  
  public void addLayer(Layer layer) {
    subLayers.add(layer);
    totalSize += layer.size();
  }

  @Override
  public void forward() {
    for(Layer l : subLayers)
      l.forward();
  }

  @Override
  public void backward() {
    for(Layer l : subLayers)
      l.backward();
  }

  @Override
  public double[] derivative() {
    double[] derivatives = new double[totalSize];
    int start = 0;
    for(Layer l : subLayers) {
      double[] d = l.derivative();
      System.arraycopy(d, 0, derivatives, start, d.length);
      start += d.length;
    }
    return derivatives;
  }

  @Override
  public double[] getOutput() {
    double[] derivatives = new double[totalSize];
    int start = 0;
    for(Layer l : subLayers) {
      double[] d = l.derivative();
      System.arraycopy(d, 0, derivatives, start, d.length);
      start += d.length;
    }
    return derivatives;
  }

  @Override
  public double[] getGradient() {
    // TODO Auto-generated method stub
    return null;
  }
  
  public List<Layer> getSubLayers() {
    return subLayers;
  }

  @Override
  public NetworkElement[] getElements() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public void addNetworkElements(NetworkElement... elements) {
    // TODO Auto-generated method stub

  }

  @Override
  public int size() {
    // TODO Auto-generated method stub
    return 0;
  }

}
