// app.js
App({
  globalData: {
    // 修改为你的开发电脑局域网IP
    apiBaseUrl: "http://172.16.6.110:5000",  // 替换为实际IP
    userInfo: null
  },
  
  onLaunch() {
    console.log('小程序启动成功');
    this.getUserInfo();
  },
  
  getUserInfo() {
    // 获取用户信息示例
    wx.getSetting({
      success: res => {
        if (res.authSetting['scope.userInfo']) {
          // 已经授权，可以直接调用 getUserInfo 获取头像昵称
          wx.getUserInfo({
            success: res => {
              this.globalData.userInfo = res.userInfo;
              console.log('用户信息:', res.userInfo);
            }
          });
        }
      }
    });
  }
});    