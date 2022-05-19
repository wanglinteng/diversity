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
	"time"
)

func Dpp(rank_scores []float32, item_dssms [][]float32, sampling_count int, epsilon float32) ([]int, error){
	/**
	    DPP (Determinantal Point Process)

		输入参数
			rank_scores: 精排模型打分、排重等操作后分数, 由大->小排序
			item_dssms: item的向量，与rank_scores顺序一致
			sampling_count: DPP采样的条数
			epsilon: DPP计算参数，建议设置为0.01

		输出结果
			rank_scores 采样索引顺序，默认 {0, 1, ... , len(sampling_count)}

	**/
	item_count := len(rank_scores)
	if len(item_dssms) != item_count || item_count < sampling_count {
		return nil, fmt.Errorf("err %s", "len(item_dssms) != item_count or item_count < sampling_count")
	}
	scores := tensor.New(tensor.WithShape(item_count, 1), tensor.WithBacking(rank_scores))
	// fmt.Printf("scores %v:\n%v\n", scores.Shape(), scores) 
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
	score_r_1, _ := tensor.Repeat(scores, 1, item_count)
	score_r_2, _ := tensor.T(score_r_1)
	kernel_matrix_0, _ := tensor.Mul(score_r_1, sim_matrix)
	kernel_matrix, _ := tensor.Mul(kernel_matrix_0, score_r_2)
	// fmt.Printf("kernel_matrix:\n%v\n", kernel_matrix)
	// c
	c_0 := make([]float32, sampling_count * item_count)
	for i := range c_0 {
		c_0[i] = float32(0.0)
	}
	c := tensor.New(tensor.WithShape(sampling_count, item_count), tensor.WithBacking(c_0))
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
		var ei float32
		for i := range Z_Y {
			kji_0, _ := kernel_matrix.At(j ,i) 
			kji := kji_0.(float32)
			dj, _ := d.At(j)
			dj_sqrt := float32(math.Sqrt(float64(dj.(float32))))
			if iter == 0 { 
				ei = kji / dj_sqrt
			}else{
				c_j, _ := c.Slice(tensor.S(0, iter), tensor.S(j))
				c_i, _ := c.Slice(tensor.S(0, iter), tensor.S(i))
				cji, _ := tensor.Dot(c_j.Materialize(),  c_i.Materialize())  // Must Materialize
				ei = (kji - cji.Data().(float32)) / dj_sqrt
			}
			c.SetAt(ei, iter, i) 
			di, _ := d.At(i)
			d.SetAt(di.(float32) - ei * ei, i) 
		}

		d.SetAt(float32(0), j)
		j_0, _ = tensor.Argmax(d, 0)
		j = j_0.Data().(int)
		dx, _ := d.At(j)
		if dx.(float32) < epsilon {
			break
		}
		Yg.Insert(j)
		// fmt.Printf("Yg(iter|%v):\n%v\n", iter, Yg )
		iter += 1
		// while
		if Yg.Len() >= sampling_count {
			break
		}
	}
    // fmt.Printf("Yg:\n%v\n", Yg.List())
	return Yg.List(), nil
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
	sampling_count := int(100)
	epsilon := float32(0.01)
	uuids, scores, dssms := testData("./data.txt", 500)
	fmt.Printf("uuids:\n%v\n", len(uuids))
	// dpp
	start := time.Now()
	var dpp_rank []int
	for i := 0; i < 1000; i++ {
		dpp_rank, _ = Dpp(scores, dssms, sampling_count, epsilon)
	}
	// dpp_rank, _ = Dpp(scores, dssms, sampling_count, epsilon)
	cost := time.Since(start) / time.Millisecond 
	fmt.Printf("cost=[%dms], avg=[%vms]", cost, float32(cost) / 1000)
	fmt.Printf("dpp_rank:\n%v\n", dpp_rank)
}