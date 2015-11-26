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
		this.wordVectorSize = FeatureFactory.allVecs.numRows();
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
			SimpleMatrix X = getWordVector(wordsInWindow); //100 * 251
			if (X != null)
			{
				SimpleMatrix Z = W.mult(X); //100 * 1
				SimpleMatrix H = getTanh(Z); //101 * 1
				SimpleMatrix O = U.mult(H); //5 * 1
				SimpleMatrix P = getSoftmax(O); //5 * 1
				SimpleMatrix Y = getY(wordsInWindow); //5 * 1
				SimpleMatrix delta2 = P.minus(Y); //5 * 1
				SimpleMatrix UPrime = delta2.mult(H.transpose()); //5 * 101
				U = U.plus(UPrime.scale(alpha)); //5 * 101
				SimpleMatrix delta1 = U.transpose().mult(delta2).elementMult(getDTanh(Z));
				System.out.println("delta1: "+delta1.numRows()+" * "+delta1.numCols());
			}
		}
	}

	private SimpleMatrix getWordVector(Datum[] wordsInWindow)
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

		double[] firstVector;
		if (wordsInWindow[0].equals(Datum.SEPARATE_WORD))
		{
			firstVector = this.getWordVector(Datum.START_WORD);
		}
		else
		{
			firstVector = this.getWordVector(wordsInWindow[0].word);
		}

		double[] lastVector;
		if (wordsInWindow[wordsInWindow.length-1].equals(Datum.SEPARATE_WORD))
		{
			lastVector = this.getWordVector(Datum.END_WORD);
		}
		else
		{
			lastVector = this.getWordVector(wordsInWindow[wordsInWindow.length-1].word);
		}

		double[] finalVector = new double[wordVectorSize * wordsInWindow.length + 1];

		// Concatenate the three vectors
		for (int i=0; i<firstVector.length; i++)
		{
			finalVector[i] = firstVector[i];
		}

		for (int i=1; i<wordsInWindow.length-1; i++)
		{
			double[] middleVector = this.getWordVector(wordsInWindow[i].word);
			for (int j=0; j<middleVector.length; j++)
			{
				finalVector[wordVectorSize * i + j] = middleVector[j];
			}
		}

		for (int i=0; i<lastVector.length; i++)
		{
			finalVector[i+wordVectorSize * (wordsInWindow.length-1)] = lastVector[i];
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
		
		return getCol(FeatureFactory.allVecs, index);
	}

	private SimpleMatrix getY(Datum[] wordsInWindow)
	{
		double[] y = new double[wordsInWindow.length];
		for (int i=0; i<wordsInWindow.length; i++)
		{
			y[i] = 1; //TODO assign the value of y according to the label
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
		}
		tanh[tanh.length-1] = 1;
		double[][] tanHData = {tanh};
		return new SimpleMatrix(tanHData).transpose();
	}
	
	private SimpleMatrix getSoftmax(SimpleMatrix input)
	{
		double softMax[] = new double[input.numRows()];
		double sum = 0;
		for(int i=0; i<input.numRows(); i++)
		{
			softMax[i] = Math.exp(input.get(i, 0));
			sum += softMax[i];
		}
		for(int i=0; i<softMax.length; i++)
		{
			softMax[i] = softMax[i]/sum;
		}
		double[][] softMaxData = {softMax};
		return new SimpleMatrix(softMaxData).transpose();
	}
	
	public double[] addExtraOne(double[] input)
	{
		double[] output = new double[input.length + 1];
		for (int i=0; i<input.length; i++)
		{
			output[i] = input[i];
		}
		output[output.length-1] = 1;
		return output;
	}
	
	public double[] getRow(SimpleMatrix sm, int row)
	{
		double[] vector = new double[sm.numCols()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = sm.get(row, i);
		}
		return vector;
	}
	
	public double[] getCol(SimpleMatrix sm, int col)
	{
		double[] vector = new double[sm.numRows()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = sm.get(i, col);
		}
		return vector;
	}
	
	private void nnTest(List<Datum> testData)
	{

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
