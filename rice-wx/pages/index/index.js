Page({
  data: {
    originalImageUrl: '',  // 原始图片路径
    resultImageUrl: '',    // 结果图片路径
    diseaseName: '',       // 病害名称
    description: '',       // 病害描述
    treatment: '',         // 防治建议
    confidence: '',        // 置信度
    confidencePercent: '0%', // 置信度百分比
    severity: '',          // 严重程度
    severityClass: '',     // 严重程度CSS类
    hasImage: false,       // 是否已选择图片
    showResult: false,     // 是否显示结果
    loading: false,        // 加载状态
    errorMessage: '',      // 错误信息
    baseUrl: "http://172.16.6.110:5000",  // 后端基础URL，需替换为实际IP和端口
    showDiseaseModal: false, // 是否显示病害详情模态框
    selectedDisease: {},   // 当前选中的病害信息
    diseases: [            // 常见病害数据
      {
        id: 1,
        name: '稻瘟病',
        imageUrl: '../images/bacterial-blight.jpg',
        affectedPart: '叶片、茎秆、穗部',
        severity: '高',
        severityClass: 'text-danger',
        description: '稻瘟病是由稻瘟病菌引起的一种重要水稻病害，可危害水稻各生育期和各部位，是世界水稻生产上危害最严重的病害之一。',
        treatment: '1. 选用抗病品种；2. 加强田间管理，合理施肥；3. 化学防治：在发病初期及时喷施三环唑、稻瘟灵等药剂。'
      },
      {
        id: 2,
        name: '纹枯病',
        imageUrl: '../images/sheath_blight238.jpg',
        affectedPart: '叶鞘、叶片',
        severity: '中',
        severityClass: 'text-warning',
        description: '纹枯病是由立枯丝核菌引起的一种常见水稻病害，主要危害叶鞘和叶片，严重时可导致稻株倒伏，影响产量。',
        treatment: '1. 打捞菌核，减少菌源；2. 合理密植，改善通风透光条件；3. 化学防治：在发病初期喷施井冈霉素、噻呋酰胺等药剂。'
      },
      {
        id: 3,
        name: '白叶枯病',
        imageUrl: '../images/bacterial_leaf_blight19.jpg',
        affectedPart: '叶片',
        severity: '高',
        severityClass: 'text-danger',
        description: '白叶枯病是由黄单胞杆菌引起的一种细菌性病害，主要危害叶片，病斑呈灰白色，边缘有黄色晕圈，严重时全叶枯死。',
        treatment: '1. 选用抗病品种；2. 种子消毒；3. 及时清除病株；4. 化学防治：喷施噻菌铜、叶枯唑等药剂。'
      },
      {
        id: 4,
        name: '稻曲病',
        imageUrl: '../images/leaf_scald47.jpg',
        affectedPart: '稻穗',
        severity: '中',
        severityClass: 'text-warning',
        description: '稻曲病是由稻绿核菌引起的一种真菌病害，主要危害稻穗，形成墨绿色的稻曲球，不仅影响产量，还会产生毒素，危害人畜健康。',
        treatment: '1. 选用抗病品种；2. 合理施肥，避免偏施氮肥；3. 化学防治：在破口前5-7天喷施戊唑醇、苯醚甲环唑等药剂。'
      }
    ]
  },

  // 选择图片
  chooseImage() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],  // 压缩图
      sourceType: ['album', 'camera'],
      success: (res) => {
        const tempFilePaths = res.tempFilePaths[0];
        this.compressImage(tempFilePaths);
      }
    });
  },

  // 图片压缩处理
  compressImage(tempFilePath) {
    wx.compressImage({
      src: tempFilePath,
      quality: 50,  // 压缩质量
      success: (res) => {
        this.setData({
          originalImageUrl: res.tempFilePath,
          hasImage: true,
          showResult: false,
          errorMessage: ""
        });
      },
      fail: (err) => {
        wx.showToast({ title: "图片压缩失败，使用原图", icon: "none" });
        this.setData({
          originalImageUrl: tempFilePath,
          hasImage: true,
          showResult: false,
          errorMessage: ""
        });
      }
    });
  },

  // 上传图片并获取预测结果
  uploadImage() {
    if (!this.data.originalImageUrl) {
      wx.showToast({ title: "请先选择图片", icon: "none" });
      return;
    }

    this.setData({ loading: true, errorMessage: "" });
    
    // 后端API路径
    const apiUrl = `${this.data.baseUrl}/api/predict`;
    
    wx.uploadFile({
      url: apiUrl,
      filePath: this.data.originalImageUrl,
      name: 'image',
      timeout: 30000,  // 超时时间30秒
      success: (res) => {
        this.setData({ loading: false });
        console.log("服务器响应数据:", res.data);
        
        try {
          const result = JSON.parse(res.data);
          
          if (result.status === "success") {
            // 关键修改：只拼接基础URL和图片路径，移除多余的/api/predict
            const resultImageUrl = this.data.baseUrl + result.data.result_image_url;
            
            console.log("后端返回的图片路径:", result.data.result_image_url);
            console.log("拼接后的完整URL:", resultImageUrl);
            
            // 计算置信度百分比
            const confidenceValue = parseFloat(result.data.confidence);
            const confidencePercent = isNaN(confidenceValue) ? '0%' : `${confidenceValue * 100}%`;
            
            // 设置严重程度的CSS类
            let severityClass = '';
            if (result.data.severity === '高') {
              severityClass = 'severity-high';
            } else if (result.data.severity === '中') {
              severityClass = 'severity-medium';
            } else {
              severityClass = 'severity-low';
            }
            
            this.setData({
              showResult: true,
              resultImageUrl: resultImageUrl,
              diseaseName: result.data.disease_name,
              description: result.data.description,
              treatment: result.data.treatment,
              confidence: result.data.confidence,
              confidencePercent: confidencePercent,
              severity: result.data.severity,
              severityClass: severityClass
            });
          } else {
            this.setData({ errorMessage: result.message || "识别失败，请重试" });
          }
        } catch (e) {
          console.error("JSON解析错误:", e);
          console.error("原始响应:", res.data);
          this.setData({ 
            errorMessage: "服务器响应格式错误，请检查后端配置",
            rawResponse: res.data
          });
        }
      },
      fail: (err) => {
        this.setData({
          loading: false,
          errorMessage: "网络错误，请检查后端是否启动或IP是否正确"
        });
        console.error("上传失败：", err);
      }
    });
  },

  // 重新选择图片
  resetImage() {
    this.setData({
      originalImageUrl: '',
      resultImageUrl: '',
      hasImage: false,
      showResult: false,
      errorMessage: ''
    });
  },
  
  // 显示病害详情
  showDiseaseDetail: function(e) {
    const diseaseId = e.currentTarget.dataset.id;
    const disease = this.data.diseases.find(item => item.id === diseaseId);
    if (disease) {
      console.log("点击病害:", disease.name); // 添加日志检查
      this.setData({
        showDiseaseModal: true,
        selectedDisease: disease
      });
    }
  },
  
  // 关闭病害详情
  closeDiseaseModal: function() {
    this.setData({
      showDiseaseModal: false
    });
  }
})