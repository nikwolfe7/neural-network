package mlsp.cs.cmu.edu.structure;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

import mlsp.cs.cmu.edu.elements.NetworkElement;

public class NetworkElementLayer implements Layer {

	private NetworkElement[] elements;
	private int blockSize = 100;
	private LinkedBlockingQueue<NetworkElement> queue;
	private Command forward, backward;
	private Queue<Thread> runnables;

	public NetworkElementLayer(NetworkElement... elements) {
		this.queue = new LinkedBlockingQueue<NetworkElement>();
		this.elements = elements;
		int cores = Runtime.getRuntime().availableProcessors();
		this.blockSize = (size() / cores) <= 0 ? blockSize : (size() / cores);
		this.forward = new ForwardCommand();
		this.backward = new BackwardCommand();
		this.runnables = new LinkedList<Thread>();
	}

	private abstract class Command {
		public abstract void doCommand();
	}

	private class ForwardCommand extends Command {
		@Override
		public void doCommand() {
			try {
				queue.take().forward();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	private class BackwardCommand extends Command {
		@Override
		public void doCommand() {
			try {
				queue.take().backward();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public void forward() {
		for (int i = 0; i < elements.length; i++)
			dispatchThreadsOnCommand(elements[i], i, forward);
		joinRunnables();
	}

	@Override
	public void backward() {
		for (int i = 0; i < elements.length; i++)
			dispatchThreadsOnCommand(elements[i], i, backward);
		joinRunnables();
	}

	private void dispatchThreadsOnCommand(NetworkElement networkElement, int i, Command cmd) {
		try {
			queue.put(networkElement);
			if ((i % blockSize == 0) || (i == size() - 1)) {
				Thread r = new Thread() {
					@Override
					public void run() {
						while (!queue.isEmpty()) {
							cmd.doCommand();
						}
					}
				};
				runnables.add(r);
				r.run();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	private void joinRunnables() {
		for(Thread r : runnables) {
			try {
				r.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public double[] derivative() {
		double[] d = new double[elements.length];
		for (int i = 0; i < d.length; i++) {
			d[i] = elements[i].derivative();
		}
		return d;
	}

	@Override
	public double[] getOutput() {
		double[] o = new double[elements.length];
		for (int i = 0; i < o.length; i++) {
			o[i] = elements[i].getOutput();
		}
		return o;
	}

	@Override
	public double[] getErrorTerm() {
		double[] e = new double[elements.length];
		for (int i = 0; i < e.length; i++) {
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
