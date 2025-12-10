using System.Runtime.InteropServices;
using System.Security.Claims;
using System.Text;
using BookRent.Orchestrator.Api.Requests;
using BookRent.Orchestrator.Clients;
using BookRent.Orchestrator.Services.Interfaces;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;

namespace BookRent.Orchestrator.Api;

public static class AuthentificationSagas
{
    public static async Task<Results<Ok<string>, BadRequest>> Register([FromBody] RegisterOnStoreRequest request, IIdentityService service)
    {
        
        /*var hashedPassword = new PasswordHasher<RegisterOnStoreRequest>()
            .HashPassword(user, request.Password);*/
        return TypedResults.Ok("");

    }
    
    public static async Task<Results<Ok<string>, BadRequest>> Login([FromBody] AuthentificationRequest request, IIdentityService service)
    {
        
        string token = CreateToken(request);
        return TypedResults.Ok(token);
    }

    private static string CreateToken(AuthentificationRequest user)
    {
        var claims = new List<Claim>
        {
            new Claim(ClaimTypes.Name, user.Username),
        };
        
        /*var key = new SymmetricSecurityKey(
            Encoding.UTF8.GetBytes(Configuration.GetValue<string>("AppSettings:Token")))*/
        return "Bearer";
    }
}