package main

import (
	"fmt"
	"gorgonia.org/tensor"
	"math"
	"k8s.io/apimachinery/pkg/util/sets"
	"os"
	"bufio"
	"strings"
	"strconv"
)


func dpp(rank_scores []float32, item_dssms [][]float32, max_iter int, epsilon float32) {
	
    // ---- init ----
	// max_iter & epsilon
	fmt.Printf("max_iter:\n%v\n", max_iter)
	fmt.Printf("epsilon:\n%v\n", epsilon) 
	// scores
	item_count := len(rank_scores)
	scores := tensor.New(tensor.WithShape(item_count, 1), tensor.WithBacking(rank_scores))
	// fmt.Printf("scores %v:\n%v\n", scores.Shape(), scores) 
	// dssms
	item_dssms_count := len(item_dssms)
	if item_count != item_dssms_count {
		return
	}
	dssm_dim := len(item_dssms[0])
	item_dssms_flat := make([]float32, item_count * dssm_dim)
	k := int(0)
    for i := 0; i < item_count; i++ {
		for j := 0; j < dssm_dim; j++ {
			item_dssms_flat[k] = item_dssms[i][j]
			k = k + 1
		}
	}
    dssms := tensor.New(tensor.WithShape(item_count, dssm_dim), tensor.WithBacking(item_dssms_flat))
	// fmt.Printf("dssms %v:\n%v\n", dssms.Shape(), dssms) 

	// ---- dpp ----
	// sim_matrix
	dssms_t, _ := tensor.T(dssms)
	sim_matrix, _ := tensor.Dot(dssms, dssms_t)
	// fmt.Printf("sim_matrix:\n%v\n", sim_matrix) 
	// kernel_matrix
	scores_t, _ := tensor.T(scores)
	score_r_1, _ := tensor.Repeat(scores, 1, item_count)
	score_r_2, _ := tensor.Repeat(scores_t, 0, item_count)
	kernel_matrix_0, _ := tensor.Mul(score_r_1, sim_matrix)
	kernel_matrix, _ := tensor.Mul(kernel_matrix_0, score_r_2)
	// fmt.Printf("kernel_matrix:\n%v\n", kernel_matrix)
	// c
	c_0 := make([]float32, max_iter * item_count)
	for i := range c_0 {
		c_0[i] = float32(0.0)
	}
	c := tensor.New(tensor.WithShape(max_iter, item_count), tensor.WithBacking(c_0))
	// fmt.Printf("c:\n%v\n", c)
	// d
	d_0, _ := tensor.Diag(kernel_matrix)
	d, _ := d_0.Slice(tensor.S(0), nil)
	// fmt.Printf("d:\n%v\n", d)
    // j
	j_0, _ := tensor.Argmax(d, 0)
	j := j_0.Data().(int)
	// fmt.Printf("j:\n%v\n", j)
	// Yg
	Yg := sets.NewInt(j)
	fmt.Printf("Yg:\n%v\n", Yg)
	// iter
	iter := int(0)
	// fmt.Printf("iter:\n%v\n", iter)
	// Z
	Z := sets.NewInt()
	for i := 0; i < item_count; i++ {
		Z.Insert(i)
	}
	// fmt.Printf("Z %v:\n%v\n", Z.Len(), Z)

	for {
		Z_Y := Z.Difference(Yg)
		var ei float64
		for i := range Z_Y {
			kji_0, _ := kernel_matrix.At(j ,i) 
			kji := float64(kji_0.(float32))
			dj, _ := d.At(j)
			dj_sqrt := math.Sqrt(float64(dj.(float32)))
			
			if iter == 0 { 
				ei = kji / dj_sqrt
			}else{
				c_j, _ := c.Slice(tensor.S(0, iter), tensor.S(j))
				c_i, _ := c.Slice(tensor.S(0, iter), tensor.S(i))
				cji, _ := tensor.Dot(c_j.Materialize(),  c_i.Materialize())  // Must Materialize
				ei = (kji - float64(cji.Data().(float32))) / dj_sqrt
			}
			c.SetAt(float32(ei), iter, i) 
			di, _ := d.At(i)
			d.SetAt(float32(float64(di.(float32)) - ei * ei), i) 
		}

		d.SetAt(float32(0), j)
		j_0, _ = tensor.Argmax(d, 0)
		j = j_0.Data().(int)
		dx, _ := d.At(j)
		if dx.(float32) < epsilon {
			break
		}
		Yg.Insert(j)
		fmt.Printf("Yg(iter|%v):\n%v\n", iter, Yg )
		iter += 1
		// while
		if Yg.Len() >= max_iter {
			break
		}
	}
    fmt.Printf("Yg:\n%v\n", Yg)
}

func testData(fileName string, max_cnt int) ([]string, []float32, [][]float32) {
	file, _ := os.Open(fileName)
	var uuids []string
	var scores []float32
	var dssms [][]float32
    cnt := int(0)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		arr := strings.Split(line, "\t")
		uuid := arr[0]
		score, _ := strconv.ParseFloat(arr[1], 32)
		dssms_str := strings.Split(arr[2], ",")
		var dssm []float32
		for _, i := range dssms_str {
			if v, err := strconv.ParseFloat(i, 32); err == nil {
				dssm = append(dssm, float32(v))
			}
		}
		uuids = append(uuids, uuid)
		scores = append(scores, float32(score))
		dssms = append(dssms, dssm)
		cnt += 1
		if cnt >= max_cnt {
			break
		}
	}
	return uuids, scores, dssms
}

func main() {
	// parameter
	max_iter := int(6)
	epsilon := float32(0.01)
	uuids, scores, dssms := testData("./data.txt", 50)
	fmt.Printf("uuids:\n%v\n", uuids)
	// dpp
	dpp(scores, dssms, max_iter, epsilon)
}