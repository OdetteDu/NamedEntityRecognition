package cs224n.deep;

import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;
import java.io.*;

public class WindowModel {

	private HashMap<String, String> baselineWordMap;
	protected SimpleMatrix L, W, U;
	public int wordVectorSize;
	public int windowSize;
	public int hiddenSize;
	public double alpha;


	public WindowModel(int _windowSize, int _hiddenSize, double _lr) {
		this.baselineWordMap = new HashMap<String, String>();
		this.L = FeatureFactory.allVecs;
		this.wordVectorSize = L.numRows();
		this.windowSize = _windowSize;
		this.hiddenSize = _hiddenSize;
		this.alpha = _lr;
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		int H = hiddenSize; // fanOut = H
		int Cn = windowSize * wordVectorSize; // fanIn = nC
		int K = Datum.POSSIBLE_LABELS.length;
		double range = Math.sqrt(6) / Math.sqrt(H + Cn);
		W = SimpleMatrix.random(H, Cn + 1, -range, range, new Random()); //100 * 251
		U = SimpleMatrix.random(K, H + 1, -range, range, new Random()); //5 * 101
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> trainData) {
		this.baselineTrain(trainData);
		this.nnTrain(trainData);
	}

	public void test(List<Datum> testData) {
		this.baselineTest(testData);
		this.nnTest(testData);
	}

	private void baselineTrain(List<Datum> trainData) {
		for (Datum datum : trainData) {
			this.baselineWordMap.put(datum.word, datum.label);
		}
	}

	private void baselineTest(List<Datum> testData) {

		List<Prediction> output = new ArrayList<Prediction>();
		for (Datum datum : testData) {
			String word = datum.word;
			String predictedLabel;
			if (this.baselineWordMap.containsKey(word)) {
				predictedLabel = this.baselineWordMap.get(word);
			} else {
				predictedLabel = Datum.DEFAULT_LABEL;
			}
			output.add(new Prediction(word, datum.label, predictedLabel));
		}
		this.outputToFile("baseline.out", output);
	}

	private void nnTrain(List<Datum> trainData)
	{
		System.out.println("W: "+W.numRows()+" * "+W.numCols());
		System.out.println("U: "+U.numRows()+" * "+U.numCols());
		for (int i=windowSize-1; i<trainData.size(); i++)
		{
			Datum[] wordsInWindow = new Datum[windowSize];
			int index = 0;
			for (int j=i-(windowSize-1); j<=i; j++)
			{
				wordsInWindow[index] = trainData.get(j);
				index++;
			}
			wordsInWindow = getAdjustedWordInWindow(wordsInWindow);

			if (wordsInWindow != null)
			{
				SimpleMatrix X = getWordVector(wordsInWindow); //251 * 1
				SimpleMatrix Z = W.mult(X); //100 * 1
				SimpleMatrix H = getTanh(Z); //101 * 1
				SimpleMatrix O = U.mult(H); //5 * 1
				SimpleMatrix P = getSoftmax(O); //5 * 1
				SimpleMatrix Y = getY(wordsInWindow); //5 * 1
				SimpleMatrix delta2 = P.minus(Y); //5 * 1
				SimpleMatrix UPrime = delta2.mult(H.transpose()); //5 * 101
				U = U.plus(UPrime.scale(alpha)); //5 * 101
				if(Double.isNaN(U.get(0,0)))
				{
					System.out.println(U);
				}
				SimpleMatrix delta1 = U.transpose().mult(delta2).elementMult(getDTanh(Z)); //101 * 1
				delta1 = removeExtraRow(delta1); //100 * 1
				SimpleMatrix WPrime = delta1.mult(X.transpose()); //100 * 251
				W = W.plus(WPrime.scale(alpha)); //100 * 251
				if(Double.isNaN(W.get(0, 0)))
				{
					System.out.println(W);
				}
				SimpleMatrix XPrime = W.transpose().mult(delta1); //251 * 1
				updateWordVector(wordsInWindow, XPrime);
			}
		}
		System.out.println("Finished Training");
	}

	private void nnTest(List<Datum> testData)
	{
		List<Prediction> output = new ArrayList<Prediction>();
		for (int i=windowSize-1; i<testData.size(); i++)
		{
			Datum[] wordsInWindow = new Datum[windowSize];
			int index = 0;
			for (int j=i-(windowSize-1); j<=i; j++)
			{
				wordsInWindow[index] = testData.get(j);
				index++;
			}
			wordsInWindow = getAdjustedWordInWindow(wordsInWindow);

			if (wordsInWindow != null)
			{
				Datum word = wordsInWindow[wordsInWindow.length/2];
				if (!word.word.equals(Datum.START_WORD) && !word.word.equals(Datum.END_WORD))
				{
					SimpleMatrix X = getWordVector(wordsInWindow); //251 * 1
					SimpleMatrix Z = W.mult(X); //100 * 1
					SimpleMatrix H = getTanh(Z); //101 * 1
					SimpleMatrix O = U.mult(H); //5 * 1
					SimpleMatrix P = getSoftmax(O); //5 * 1
					String label = getLabel(P);
					output.add(new Prediction(word.word, word.label, label));
				}
			}
		}
		this.outputToFile("nn.out", output);
	}

	private String getLabel(SimpleMatrix p)
	{
		double max = 0;
		String label = null;
		for(int i=0; i<p.numRows(); i++)
		{
			if(p.get(i, 0) > max)
			{
				max = p.get(i, 0);
				label = Datum.POSSIBLE_LABELS[i];
			}
		}
		return label;
	}

	private Datum[] getAdjustedWordInWindow(Datum[] wordsInWindow)
	{
		if (wordsInWindow.length < 2)
		{
			return null;
		}

		for (int i=1; i<wordsInWindow.length-1; i++)
		{
			if (wordsInWindow[i].equals(Datum.SEPARATE_WORD))
			{
				return null;
			}
		}

		Datum[] adjustedWord = new Datum[wordsInWindow.length];
		if (wordsInWindow[0].word.equals(Datum.SEPARATE_WORD))
		{
			adjustedWord[0] = new Datum(Datum.START_WORD, Datum.DEFAULT_LABEL);
		}
		else
		{
			adjustedWord[0] = wordsInWindow[0];
		}

		if (wordsInWindow[wordsInWindow.length-1].word.equals(Datum.SEPARATE_WORD))
		{
			adjustedWord[wordsInWindow.length-1] = new Datum(Datum.END_WORD, Datum.DEFAULT_LABEL);
		}
		else
		{
			adjustedWord[wordsInWindow.length-1] = wordsInWindow[wordsInWindow.length-1];
		}

		for (int i=1; i<wordsInWindow.length-1; i++)
		{
			adjustedWord[i] = wordsInWindow[i];
		}

		return adjustedWord;
	}

	private SimpleMatrix getWordVector(Datum[] wordsInWindow)
	{
		double[] finalVector = new double[wordVectorSize * wordsInWindow.length + 1];
		for (int i=0; i<wordsInWindow.length; i++)
		{
			double[] wordVector = this.getWordVector(wordsInWindow[i].word);
			for (int j=0; j<wordVector.length; j++)
			{
				finalVector[wordVectorSize * i + j] = wordVector[j];
			}
		}
		//Bias term
		finalVector[finalVector.length-1] = 1;
		double [][] matrixData = {finalVector};
		SimpleMatrix sm = new SimpleMatrix(matrixData);
		return sm.transpose();
	}

	private double[] getWordVector(String word)
	{
		int index;
		if(FeatureFactory.wordToNum.containsKey(word))
		{
			index = FeatureFactory.wordToNum.get(word);
		}
		else
		{
			index = FeatureFactory.NON_EXISTING_VOCAB_INDEX;
		}

		return getCol(L, index);
	}

	private void updateWordVector(Datum[] wordsInWindow, SimpleMatrix vector)
	{
		int vectorIndex = 0;
		for (int i=0; i<wordsInWindow.length; i++)
		{
			int colIndex;
			if (FeatureFactory.wordToNum.containsKey(wordsInWindow[i]))
			{
				colIndex = FeatureFactory.wordToNum.get(wordsInWindow[i].word);
			}
			else
			{
				colIndex = FeatureFactory.NON_EXISTING_VOCAB_INDEX;
			}
			for (int j=0; j<this.wordVectorSize; j++)
			{
				double originValue = L.get(j, colIndex);
				L.set(j, colIndex, originValue + alpha * vector.get(vectorIndex, 0));
				vectorIndex ++;
			}
		}
	}

	private SimpleMatrix getY(Datum[] wordsInWindow)
	{
		double[] y = new double[Datum.POSSIBLE_LABELS.length];
		String currLabel = wordsInWindow[wordsInWindow.length/2].label;
		for (int i=0; i<y.length; i++)
		{
			if(currLabel.equals(Datum.POSSIBLE_LABELS[i]))
			{
				y[i] = 1; 
			}
		}
		double[][] yData = {y};
		return new SimpleMatrix(yData).transpose();
	}

	private SimpleMatrix getTanh(SimpleMatrix input)
	{
		double[] tanh = new double[input.numRows() + 1];
		for (int i=0; i<input.numRows(); i++)
		{
			tanh[i] = Math.tanh(input.get(i, 0));
			if(Double.isNaN(tanh[i]))
			{
				System.out.println(input);
			}
		}
		tanh[tanh.length-1] = 1;
		double[][] tanHData = {tanh};
		return new SimpleMatrix(tanHData).transpose();
	}

	private SimpleMatrix getDTanh(SimpleMatrix input)
	{
		double[] tanh = new double[input.numRows() + 1];
		for (int i=0; i<input.numRows(); i++)
		{
			tanh[i] = Math.tanh(input.get(i, 0));
			tanh[i] = 1 - tanh[i] * tanh[i];
			if(Double.isNaN(tanh[i]))
			{
				System.out.println(input);
			}
		}
		tanh[tanh.length-1] = 1;
		double[][] tanHData = {tanh};
		return new SimpleMatrix(tanHData).transpose();
	}

	private SimpleMatrix getSoftmax(SimpleMatrix input)
	{
		double[] softMax = new double[input.numRows()];
		double max = 0;
		for(int i=0; i<input.numRows(); i++)
		{
			if(input.get(i, 0) > max)
			{
				max = input.get(i, 0);
			}
		}
		double sum = 0;
		for(int i=0; i<input.numRows(); i++)
		{
			softMax[i] = Math.exp(input.get(i, 0) - max);
			sum += softMax[i];
		}
		double[] output = new double[input.numRows()];
		for(int i=0; i<softMax.length; i++)
		{
			output[i] = softMax[i]/sum;
			if(Double.isNaN(output[i]))
			{
				System.out.println(input);
			}
		}
		double[][] softMaxData = {output};
		return new SimpleMatrix(softMaxData).transpose();
	}

	private SimpleMatrix removeExtraRow(SimpleMatrix input)
	{
		double[] output = new double[input.numRows() - 1];
		for (int i=0; i<input.numRows() - 1; i++)
		{
			output[i] = input.get(i, 0);
		}
		double[][] outputData = {output};
		return new SimpleMatrix(outputData).transpose();
	}

	private double[] getRow(SimpleMatrix sm, int row)
	{
		double[] vector = new double[sm.numCols()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = sm.get(row, i);
		}
		return vector;
	}

	private double[] getCol(SimpleMatrix sm, int col)
	{
		double[] vector = new double[sm.numRows()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = sm.get(i, col);
		}
		return vector;
	}

	public void outputToFile(String fileName, List<Prediction> predictions) {

		String outputContent = "";
		for (Prediction prediction : predictions)
		{
			outputContent += prediction + "\n";
		}

		try 
		{
			File file = new File(fileName);
			if (!file.exists()) 
			{
				file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(outputContent);
			bw.close();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}

}
