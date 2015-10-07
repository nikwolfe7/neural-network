package mlsp.cs.cmu.edu.structure;

import java.util.concurrent.LinkedBlockingQueue;

import mlsp.cs.cmu.edu.elements.NetworkElement;

public class NetworkElementLayer implements Layer {

	private NetworkElement[] elements;
	private int blockSize = 25;
	private LinkedBlockingQueue<NetworkElement> queue;

	public NetworkElementLayer(NetworkElement... elements) {
	  this.queue = new LinkedBlockingQueue<NetworkElement>();
	  this.elements = elements;
	}
	
	@Override
	public void forward() {
		for(int i = 0; i < elements.length; i++)
			dispatchThreadsOnForward(elements[i], i);
	}
		
  private void dispatchThreadsOnForward(NetworkElement networkElement, int i) {
    try {
      queue.put(networkElement);
      if (i % blockSize == 1 || i > (elements.length - blockSize - 1)) {
        new Runnable() {
          @Override
          public void run() {
            while(!queue.isEmpty()) {
              try {
                queue.take().forward();
              } catch (InterruptedException e) {
                e.printStackTrace();
              }
            }
          }
        }.run();
      }
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  @Override
	public void backward() {
		for(int i = 0; i < elements.length; i++)
			elements[i].backward();
	}

	@Override
	public double[] derivative() {
		double[] d = new double[elements.length];
		for(int i = 0; i < d.length; i++) {
			d[i] = elements[i].derivative();
		}
		return d;
	}

	@Override
	public double[] getOutput() {
		double[] o = new double[elements.length];
		for(int i = 0; i < o.length; i++) {
			o[i] = elements[i].getOutput();
		}
		return o;
	}

	@Override
	public double[] getErrorTerm() {
		double[] e = new double[elements.length];
		for(int i = 0; i < e.length; i++) {
			e[i] = elements[i].getErrorTerm();
		}
		return e;
	}

	@Override
	public NetworkElement[] getElements() {
		return elements;
	}

	@Override
	public int size() {
		return elements.length;
	}

  @Override
  public void addNetworkElements(NetworkElement... newElements) {
    NetworkElement[] newElementArray = new NetworkElement[elements.length + newElements.length];
    System.arraycopy(elements, 0, newElementArray, 0, elements.length);
    System.arraycopy(newElements, 0, newElementArray, elements.length, newElements.length);
    elements = newElementArray;
  }

}
