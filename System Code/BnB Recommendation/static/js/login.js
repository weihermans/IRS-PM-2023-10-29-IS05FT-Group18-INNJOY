var login_state = false;
var signup_state = false;


function cambiar_login() {
  function displayErrorMessage(message) {
    // 将错误消息设置为元素的文本内容
    var errorMessageElement = document.getElementById('login-error-message');
    errorMessageElement.textContent = message;
    }
if (login_state){
  // 获取用户输入的邮箱和密码
  var emailOrUsername = document.querySelector('.cont_form_login input[type="text"]').value;
  var password = document.querySelector('.cont_form_login input[type="password"]').value;
      // 检查用户名文本框
      if (emailOrUsername === "") {
        displayErrorMessage("Email/Username cannot be empty");
    } else
    // 检查密码文本框
    if (password === "") {
      displayErrorMessage("Password cannot be empty");
    } 
    if (emailOrUsername !== "" && password !== "") {
        // 执行登录逻辑
        // 构造要发送的数据
  var data = {
    action: "login",
    emailOrUsername: emailOrUsername,
    password: password
  };
  
  // 发送POST请求给后端
  fetch('/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {
    if (data.message === 'Login success') {
      alert('User login successfully!');
      window.location.href = '/recommends';
      // 在这里可以跳转到用户的个人页面或其他页面
    } else {
      displayErrorMessage(data.message);
      // 在这里可以显示错误提示给用户
    }
  })
  .catch(error => {
    // 处理错误
    console.error('Error:', error);
  });
    }
  
} else{
  login_state =true;
  signup_state = false;
document.querySelector('.cont_forms').className = "cont_forms cont_forms_active_login";  
document.querySelector('.cont_form_login').style.display = "block";
document.querySelector('.cont_form_sign_up').style.opacity = "0";               

setTimeout(function(){  document.querySelector('.cont_form_login').style.opacity = "1"; },400);  
  
setTimeout(function(){    
document.querySelector('.cont_form_sign_up').style.display = "none";
},200);  
}
}

function cambiar_sign_up() {
  function displayErrorMessage(message) {
    // 将错误消息设置为元素的文本内容
    var errorMessageElement = document.getElementById('signup-error-message');
    errorMessageElement.textContent = message;
    }
  if (signup_state){
    var email = document.querySelector('.cont_form_sign_up input[placeholder="Email"]').value;
    var username = document.querySelector('.cont_form_sign_up input[placeholder="User"]').value;
    var password = document.querySelector('.cont_form_sign_up input[placeholder="Password"]').value;
    var repassword = document.querySelector('.cont_form_sign_up input[placeholder="Confirm Password"]').value;
    if (email === "") {
      displayErrorMessage("Email cannot be empty");
  } else
  // 检查密码文本框
  if (username === "") {
    displayErrorMessage("Username cannot be empty");
  } else
  if (password === "") {
    displayErrorMessage("Password cannot be empty");
  } else{

    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            action: "signup",
            email: email,
            username: username,
            password: password,
            repassword: repassword
        }),
    })
    .then(response => response.json())
    .then(data => {
        // 处理后端返回的数据
        if (data.message === 'Registration successful!') {
            alert('Registration successful!');
            window.location.href = '/recommends';
            // 在这里可以跳转到用户的个人页面或其他页面
        } else {
            displayErrorMessage(data.message);
            // 在这里可以显示错误提示给用户
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
  }
  }else {
    login_state = false;
  signup_state = true;
  document.querySelector('.cont_forms').className = "cont_forms cont_forms_active_sign_up";
  document.querySelector('.cont_form_sign_up').style.display = "block";
document.querySelector('.cont_form_login').style.opacity = "0";
  
setTimeout(function(){  document.querySelector('.cont_form_sign_up').style.opacity = "1";
},100);  

setTimeout(function(){   document.querySelector('.cont_form_login').style.display = "none";
},400);  
}

}    



function ocultar_login_sign_up() {

  login_state = false;
  signup_state = false;
  document.querySelector('.cont_forms').className = "cont_forms";  
  document.querySelector('.cont_form_sign_up').style.opacity = "0";               
  document.querySelector('.cont_form_login').style.opacity = "0"; 
  
  setTimeout(function(){
  document.querySelector('.cont_form_sign_up').style.display = "none";
  document.querySelector('.cont_form_login').style.display = "none";
  },500);  
}  
