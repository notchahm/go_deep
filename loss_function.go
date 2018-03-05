package go_deep

type LossFunction func(Vector, Vector, Vector) ValueType

func CrossEntropyLoss(label Vector, output Vector, delta_loss Vector) ValueType {
	//cross entropy loss: SUM - t ln y
	var loss ValueType = 0.0
	for index := 0; index < output.GetSize(); index++ {
		delta_loss.SetValue(index, label.GetValue(index) - output.GetValue(index))
		if label.GetValue(index) > 0 {
			if output.GetValue(index) > 0 {
				loss -= label.GetValue(index) * log(output.GetValue(index))
			} else {
				loss -= label.GetValue(index) * log(SmallestNonzeroValueType)
			}
		}
	}
	return loss
}

