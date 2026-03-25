from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	             path("UserLogin.html", views.UserLogin, name="UserLogin"),
				 path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
				 path("Register.html", views.Register, name="Register"),
				 path("SignupAction", views.SignupAction, name="SignupAction"),
				 path("AdminScreen.html", views.AdminScreen, name="AdminScreen"),
				 path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
				 path("TPredict.html", views.TPredict, name="TPredict"),
				 path("CPredict.html", views.CPredict, name="CPredict"),
				 path("AdoPredict.html", views.AdoPredict, name="AdoPredict"),
				 path("AduPredict.html", views.AduPredict, name="AduPredict"),
		     path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
		     path("LoadDataset", views.LoadDataset, name="LoadDataset"),
		     path("RunScaling", views.RunScaling, name="RunScaling"),
		     path("RunML", views.RunML, name="RunML"),
		     path("Predict", views.Predict, name="Predict"),
		     path("PredictAction", views.PredictAction, name="PredictAction"),		     	     
		    ]