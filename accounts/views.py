from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView

# Create your views here.

class RedirectAuthenticatedUserLoginView(LoginView):
    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('dashboard')
        return super().dispatch(request, *args, **kwargs)


def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('registration_complete')
    else:
        form = UserCreationForm()
    return render(request, 'accounts/register.html', {'form': form})


def password_reset(request):
    if request.method == 'POST':
        username = request.POST['username']
        new_password = request.POST['password']

        try:
            user = User.objects.get(username=username)

            if user.check_password(new_password):
                messages.error(request, "New password cannot be the same as the old password.")
            else:
                user.set_password(new_password)
                user.save()
                return redirect('password_reset_complete')
        except User.DoesNotExist:
            messages.error(request, "No account found with that username.")
    
    return render(request, 'accounts/password_reset.html')



