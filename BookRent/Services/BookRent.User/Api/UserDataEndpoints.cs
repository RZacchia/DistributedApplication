using BookRent.User.Infrastructure.Interfaces;
using BookRent.User.Models;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using UserDetailsRequest = BookRent.User.DTOs.UserDetailsRequest;
using UserDetailsResponse = BookRent.User.DTOs.UserDetailsResponse;

namespace BookRent.User.Api;

internal static class UserDataEndpoints
{
    internal static async Task<Results<Ok<UserDetailsResponse>, NotFound>> GetUser(Guid id, IUserRepository repo)
    {
        var user = await repo.GetUserDataAsync(id);
        if (user is null) return TypedResults.NotFound();

        UserDetailsResponse response = new UserDetailsResponse(user.UserId,
            user.UserName,
            user.Email,
            user.FirstName,
            user.LastName);
            
        
        return TypedResults.Ok(response);
    }

    internal static async Task<Results<Ok<UserBaseData>, BadRequest>> AddUserDetails([FromBody] UserDetailsRequest request, IUserRepository repo)
    {
        var user = new UserBaseData
        {
            UserName = request.UserName,
            FirstName = request.FirstName,
            LastName = request.LastName,
            Email = request.Email,
            UserId = request.UserId,
        };
        var result = await repo.AddUserDataAsync(user);
        if(result) return TypedResults.Ok(user); 
        return TypedResults.BadRequest();
    }
}