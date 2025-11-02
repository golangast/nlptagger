package nn

import (
	"math"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

// Optimizer interface defines the contract for optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
}

// Adam represents the Adam optimizer.
type Adam struct {
	parameters []*Tensor
	learningRate float64
	beta1      float64
	beta2      float64
	epsilon    float64
	t          int
	m          map[*Tensor]*Tensor // 1st moment vector
	v          map[*Tensor]*Tensor // 2nd moment vector
	clipValue  float64
}

// NewOptimizer creates a new Adam optimizer.
func NewOptimizer(parameters []*Tensor, learningRate float64, clipValue float64) Optimizer {
	return &Adam{
		parameters:   parameters,
		learningRate: learningRate,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-8,
		t:            0,
		m:            make(map[*Tensor]*Tensor),
		v:            make(map[*Tensor]*Tensor),
		clipValue:    clipValue,
	}
}

// Step performs a single optimization step.
func (o *Adam) Step() {
	o.t++
	for _, p := range o.parameters {
		if p.Grad != nil {
			if _, ok := o.m[p]; !ok {
				o.m[p] = NewTensor(p.Shape, make([]float64, len(p.Data)), false)
				o.v[p] = NewTensor(p.Shape, make([]float64, len(p.Data)), false)
			}

			// Clip gradients
			for i := range p.Grad.Data {
				if p.Grad.Data[i] > o.clipValue {
					p.Grad.Data[i] = o.clipValue
				} else if p.Grad.Data[i] < -o.clipValue {
					p.Grad.Data[i] = -o.clipValue
				}
			}

			// Update biased first moment estimate
			for i := range o.m[p].Data {
				o.m[p].Data[i] = o.beta1*o.m[p].Data[i] + (1-o.beta1)*p.Grad.Data[i]
			}

			// Update biased second raw moment estimate
			for i := range o.v[p].Data {
				o.v[p].Data[i] = o.beta2*o.v[p].Data[i] + (1-o.beta2)*math.Pow(p.Grad.Data[i], 2)
			}

			// Compute bias-corrected first moment estimate
			mHatData := make([]float64, len(o.m[p].Data))
			for i := range mHatData {
				mHatData[i] = o.m[p].Data[i] / (1 - math.Pow(o.beta1, float64(o.t)))
			}

			// Compute bias-corrected second raw moment estimate
			vHatData := make([]float64, len(o.v[p].Data))
			for i := range vHatData {
				vHatData[i] = o.v[p].Data[i] / (1 - math.Pow(o.beta2, float64(o.t)))
			}

			// Update parameters
			for i := range p.Data {
				p.Data[i] -= o.learningRate * mHatData[i] / (math.Sqrt(vHatData[i]) + o.epsilon)
			}
		}
	}
}

// ZeroGrad resets the gradients of all parameters.
func (o *Adam) ZeroGrad() {
	for _, p := range o.parameters {
		p.ZeroGrad()
	}
}